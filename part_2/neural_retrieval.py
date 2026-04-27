"""Neural retrieval index for Part 2 (ACL anthology).

Default model: sentence-transformers/all-mpnet-base-v2
Default fields: title + abstract
"""

from typing import Callable

import numpy as np
from sentence_transformers import SentenceTransformer


class NeuralIndex:
    """Dense retriever backed by a sentence-transformer encoder."""

    def __init__(
        self,
        model_name="sentence-transformers/all-mpnet-base-v2",
        fields=("title", "abstract"),
        device=None,
        text_builder: Callable[[dict], str] | None = None,
        encoder=None,
        batch_size=64,
    ):
        self.model_name = model_name
        self.fields = tuple(fields)
        self.text_builder = text_builder
        self.batch_size = batch_size

        if encoder is not None:
            self.encoder = encoder
        elif device is None:
            self.encoder = SentenceTransformer(model_name)
        else:
            self.encoder = SentenceTransformer(model_name, device=device)

        self.dataset = None
        self.doc_texts = None
        self.doc_embeds = None

    @staticmethod
    def _sanitize_matrix_embeddings(matrix, label):
        """Ensure 2D embeddings are finite and L2-normalized row-wise."""
        matrix = np.asarray(matrix, dtype=np.float32)
        if matrix.ndim != 2:
            raise ValueError(f"{label} must be a 2D array, got shape {matrix.shape}")

        non_finite_mask = ~np.isfinite(matrix)
        if np.any(non_finite_mask):
            bad_rows = np.where(np.any(non_finite_mask, axis=1))[0]
            print(
                f"[NeuralIndex] warning: {len(bad_rows)} {label} rows contain "
                "non-finite values; replacing nan/inf with 0.0"
            )
            matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        invalid_norm_mask = (~np.isfinite(norms)) | (norms <= 0.0)
        if np.any(invalid_norm_mask):
            bad_rows = np.where(invalid_norm_mask[:, 0])[0]
            print(
                f"[NeuralIndex] warning: {len(bad_rows)} {label} rows have "
                "zero/non-finite norm; forcing those rows to zero"
            )
            safe_norms = norms.copy()
            safe_norms[invalid_norm_mask] = 1.0
            matrix = matrix / safe_norms
            matrix[bad_rows] = 0.0
        else:
            matrix = matrix / norms

        if not np.all(np.isfinite(matrix)):
            print(
                f"[NeuralIndex] warning: non-finite values remain in {label} "
                "after normalization; applying final nan_to_num"
            )
            matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

        return np.asarray(matrix, dtype=np.float32)

    @staticmethod
    def _sanitize_vector_embedding(vector, label):
        """Ensure 1D embedding is finite and L2-normalized."""
        vector = np.asarray(vector, dtype=np.float32).reshape(-1)

        if not np.all(np.isfinite(vector)):
            bad = int(np.count_nonzero(~np.isfinite(vector)))
            print(
                f"[NeuralIndex] warning: {bad} non-finite values in {label}; "
                "replacing nan/inf with 0.0"
            )
            vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 0.0:
            print(
                f"[NeuralIndex] warning: {label} has zero/non-finite norm; "
                "using all-zero vector"
            )
            return np.zeros_like(vector, dtype=np.float32)

        vector = vector / norm

        if not np.all(np.isfinite(vector)):
            print(
                f"[NeuralIndex] warning: non-finite values remain in {label} "
                "after normalization; applying final nan_to_num"
            )
            vector = np.nan_to_num(vector, nan=0.0, posinf=0.0, neginf=0.0)

        return np.asarray(vector, dtype=np.float32)

    def _doc_to_text(self, doc):
        if self.text_builder is not None:
            text = self.text_builder(doc)
            return str(text or "")

        parts = []
        for field in self.fields:
            value = doc.get(field, "")
            if value is None:
                continue
            if isinstance(value, list):
                value = " ".join(str(x) for x in value)
            value = str(value).strip()
            if value:
                parts.append(value)
        return "\n".join(parts)

    def build(self, dataset, batch_size=None, show_progress_bar=True):
        """Encode all documents from a Hugging Face Dataset."""
        self.dataset = dataset
        self.doc_texts = [self._doc_to_text(doc) for doc in dataset]

        embeds = self.encoder.encode(
            self.doc_texts,
            batch_size=batch_size or self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,
        )
        self.doc_embeds = self._sanitize_matrix_embeddings(embeds, "document embedding")
        return self

    def encode_query(self, query):
        """Encode one query to a normalized dense vector [H]."""
        if self.doc_embeds is None:
            raise RuntimeError("Index not built. Call build(dataset) first.")

        if not query or not str(query).strip():
            return np.zeros(self.doc_embeds.shape[1], dtype=np.float32)

        vec = self.encoder.encode(
            [str(query)],
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )[0]
        return self._sanitize_vector_embedding(vec, "query embedding")

    def retrieve(self, query, k=10, threshold=None):
        """Return top-k list of (cosine_score, doc_index)."""
        if self.doc_embeds is None:
            raise RuntimeError("Index not built. Call build(dataset) first.")

        query_vec = self.encode_query(query)
        if not np.any(query_vec):
            return []

        if not np.all(np.isfinite(self.doc_embeds)):
            print(
                "[NeuralIndex] warning: document embeddings became non-finite "
                "before ranking; sanitizing"
            )
            self.doc_embeds = self._sanitize_matrix_embeddings(
                self.doc_embeds, "document embedding"
            )

        if not np.all(np.isfinite(query_vec)):
            print(
                "[NeuralIndex] warning: query embedding became non-finite "
                "before ranking; sanitizing"
            )
            query_vec = self._sanitize_vector_embedding(query_vec, "query embedding")

        # Use a stable scoring path directly to avoid backend-specific matmul
        # warnings seen on some BLAS/accelerator configurations.
        doc = np.nan_to_num(self.doc_embeds, nan=0.0, posinf=0.0, neginf=0.0)
        qry = np.nan_to_num(query_vec, nan=0.0, posinf=0.0, neginf=0.0)
        doc = np.clip(doc, -1.0, 1.0)
        qry = np.clip(qry, -1.0, 1.0)
        scores = np.einsum(
            "ij,j->i",
            doc.astype(np.float64, copy=False),
            qry.astype(np.float64, copy=False),
            optimize=True,
        )
        scores = np.asarray(scores, dtype=np.float32)
        if not np.all(np.isfinite(scores)):
            bad = int(np.count_nonzero(~np.isfinite(scores)))
            print(
                f"[NeuralIndex] warning: {bad} non-finite retrieval scores; "
                "setting them to -inf"
            )
            scores = np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)

        if threshold is not None:
            keep = np.where(scores >= threshold)[0]
            if keep.size == 0:
                return []
            ranked = keep[np.argsort(-scores[keep])]
        else:
            ranked = np.argsort(-scores)

        top = ranked[:k]
        return [(float(scores[idx]), int(idx)) for idx in top]

    def acl_ids(self, indices):
        """Map numeric document indices back to ACL-id strings."""
        if self.dataset is None:
            raise RuntimeError("Index not built. Call build(dataset) first.")
        return [str(self.dataset[int(idx)].get("acl_id", "")) for idx in indices]
