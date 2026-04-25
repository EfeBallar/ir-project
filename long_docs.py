"""Long-document retrieval helpers for Part 2.

Implements:
  - overlapping chunking of full papers with doc-level max pooling
  - structure-aware chunking using section headings, then paragraphs, then sentence-group fallback, with top-3 weighted aggregation
"""

import re

import numpy as np

from neural_retrieval import NeuralIndex


def _paper_to_full_text(paper):
    title = str(paper.get("title", "") or "").strip()
    abstract = str(paper.get("abstract", "") or "").strip()
    full_text = str(paper.get("full_text", "") or "").strip()
    return "\n".join(part for part in (title, abstract, full_text) if part)


def _paper_full_text_or_fallback(paper):
    full_text = str(paper.get("full_text", "") or "").strip()
    if full_text:
        return full_text
    return _paper_to_full_text(paper)


def chunk_paper(paper, max_words=220, overlap=40):
    """Split a paper into overlapping word windows."""
    if max_words <= 0:
        raise ValueError("max_words must be > 0")
    if overlap < 0:
        raise ValueError("overlap must be >= 0")
    if overlap >= max_words:
        raise ValueError("overlap must be smaller than max_words")

    acl_id = str(paper.get("acl_id", ""))
    text = _paper_full_text_or_fallback(paper)
    words = text.split()

    if not words:
        return []

    step = max_words - overlap
    chunks = []
    start = 0
    chunk_idx = 0

    while start < len(words):
        end = min(start + max_words, len(words))
        chunk_words = words[start:end]
        if not chunk_words:
            break
        chunks.append(
            {
                "acl_id": acl_id,
                "chunk_idx": chunk_idx,
                "text": " ".join(chunk_words),
            }
        )
        if end >= len(words):
            break
        start += step
        chunk_idx += 1

    return chunks


SECTION_HEADING_RE = re.compile(
    r"(?:(?<=^)|(?<=[\.!?]\s))(?:\d+(?:\.\d+)*)?\s*"
    r"(abstract|introduction|background|related work|method(?:ology)?|approach|"
    r"experiments?|results?|discussion|conclusion|references)\b",
    flags=re.IGNORECASE,
)


def _split_by_section_headings(text):
    starts = [match.start() for match in SECTION_HEADING_RE.finditer(text)]
    if not starts:
        return []
    if starts[0] != 0:
        starts = [0] + starts

    units = []
    for i, start in enumerate(starts):
        end = starts[i + 1] if i + 1 < len(starts) else len(text)
        unit = text[start:end].strip()
        if unit:
            units.append(unit)
    return units if len(units) >= 2 else []


def _split_by_paragraphs(text):
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    return paragraphs if len(paragraphs) >= 2 else []


def _split_by_sentence_groups(text, target_words=180, max_words=220):
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]
    if not sentences:
        return []

    groups = []
    current = []
    current_words = 0

    for sentence in sentences:
        sent_words = sentence.split()
        if not sent_words:
            continue

        if current and current_words + len(sent_words) > max_words:
            groups.append(" ".join(current).strip())
            current = []
            current_words = 0

        current.append(sentence)
        current_words += len(sent_words)

        if current_words >= target_words:
            groups.append(" ".join(current).strip())
            current = []
            current_words = 0

    if current:
        groups.append(" ".join(current).strip())

    if groups:
        return [g for g in groups if g]

    words = text.split()
    fallback = []
    for start in range(0, len(words), max_words):
        chunk = " ".join(words[start : start + max_words]).strip()
        if chunk:
            fallback.append(chunk)
    return fallback


def _split_long_unit(unit, max_words):
    words = unit.split()
    if len(words) <= max_words:
        return [unit.strip()] if unit.strip() else []
    pieces = []
    for start in range(0, len(words), max_words):
        piece = " ".join(words[start : start + max_words]).strip()
        if piece:
            pieces.append(piece)
    return pieces


def _normalize_units(units, min_words=80, max_words=220):
    cleaned = [u.strip() for u in units if u and u.strip()]
    if not cleaned:
        return []

    split_units = []
    for unit in cleaned:
        split_units.extend(_split_long_unit(unit, max_words=max_words))

    merged = []
    buffer_words = []
    for unit in split_units:
        words = unit.split()
        if not words:
            continue

        if len(words) < min_words:
            buffer_words.extend(words)
            if len(buffer_words) >= min_words:
                merged.append(" ".join(buffer_words).strip())
                buffer_words = []
            continue

        if buffer_words:
            combined = buffer_words + words
            if len(combined) <= max_words:
                merged.append(" ".join(combined).strip())
                buffer_words = []
            else:
                merged.append(" ".join(buffer_words).strip())
                buffer_words = []
                merged.append(" ".join(words).strip())
        else:
            merged.append(" ".join(words).strip())

    if buffer_words:
        if merged:
            last = merged.pop()
            combined = (last + " " + " ".join(buffer_words)).strip()
            merged.extend(_split_long_unit(combined, max_words=max_words))
        else:
            merged.append(" ".join(buffer_words).strip())

    return [u for u in merged if u]


def chunk_paper_structured(paper, max_words=220, min_words=80, target_words=180):
    """Split a paper by section/paragraph structure with sentence fallback."""
    acl_id = str(paper.get("acl_id", ""))
    text = _paper_full_text_or_fallback(paper)
    if not text.strip():
        return []

    units = _split_by_section_headings(text)
    source = "section"

    if len(units) < 2:
        units = _split_by_paragraphs(text)
        source = "paragraph"

    if len(units) < 2:
        units = _split_by_sentence_groups(text, target_words=target_words, max_words=max_words)
        source = "sentence_group"

    units = _normalize_units(units, min_words=min_words, max_words=max_words)
    if not units:
        return []

    chunks = []
    for i, unit in enumerate(units):
        chunks.append(
            {
                "acl_id": acl_id,
                "chunk_idx": i,
                "text": unit,
                "source": source,
            }
        )
    return chunks


class ChunkedIndex(NeuralIndex):
    """Dense retrieval over chunk embeddings with document-level max pooling."""

    def __init__(
        self,
        model_name="sentence-transformers/all-mpnet-base-v2",
        fields=("title", "abstract", "full_text"),
        device=None,
        text_builder=None,
        encoder=None,
        batch_size=64,
        max_words=220,
        overlap=40,
    ):
        super().__init__(
            model_name=model_name,
            fields=fields,
            device=device,
            text_builder=text_builder,
            encoder=encoder,
            batch_size=batch_size,
        )
        self.max_words = max_words
        self.overlap = overlap

        self.chunk_records = []
        self.chunk_to_doc = []
        self.doc_to_chunks = {}
        self.acl_id_to_doc_idx = {}
        self.chunk_embeds = None

    def build(self, dataset, batch_size=None, show_progress_bar=True):
        """Chunk each document, encode chunks, and build chunk/doc mappings."""
        self.dataset = dataset
        self.acl_id_to_doc_idx = {}
        self.chunk_records = []
        self.chunk_to_doc = []
        self.doc_to_chunks = {}

        for doc_idx, doc in enumerate(dataset):
            acl_id = str(doc.get("acl_id", ""))
            self.acl_id_to_doc_idx[acl_id] = doc_idx

            chunks = chunk_paper(doc, max_words=self.max_words, overlap=self.overlap)
            if not chunks:
                fallback = str(doc.get("title", "") or "").strip()
                if fallback:
                    chunks = [{"acl_id": acl_id, "chunk_idx": 0, "text": fallback}]

            chunk_indices = []
            for chunk in chunks:
                chunk_indices.append(len(self.chunk_records))
                self.chunk_records.append(chunk)
                self.chunk_to_doc.append(doc_idx)
            self.doc_to_chunks[doc_idx] = chunk_indices

        chunk_texts = [chunk["text"] for chunk in self.chunk_records]
        self.doc_texts = chunk_texts

        embeds = self.encoder.encode(
            chunk_texts,
            batch_size=batch_size or self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,
        )

        self.chunk_embeds = self._sanitize_matrix_embeddings(embeds, "chunk embedding")
        self.doc_embeds = self.chunk_embeds
        return self

    def _score_chunks(self, query_vec):
        doc = np.nan_to_num(self.chunk_embeds, nan=0.0, posinf=0.0, neginf=0.0)
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
            scores = np.nan_to_num(scores, nan=-np.inf, posinf=-np.inf, neginf=-np.inf)
        return scores

    def retrieve(self, query, k=10, threshold=None):
        """Return top-k (score, doc_idx), where score is max chunk score per doc."""
        if self.chunk_embeds is None:
            raise RuntimeError("Chunked index not built. Call build(dataset) first.")

        query_vec = self.encode_query(query)
        if not np.any(query_vec):
            return []

        chunk_scores = self._score_chunks(query_vec)
        doc_best = {}

        for chunk_idx, score in enumerate(chunk_scores):
            doc_idx = self.chunk_to_doc[chunk_idx]
            current = doc_best.get(doc_idx)
            if current is None or score > current:
                doc_best[doc_idx] = float(score)

        ranked = sorted(doc_best.items(), key=lambda item: item[1], reverse=True)
        if threshold is not None:
            ranked = [(doc_idx, score) for doc_idx, score in ranked if score >= threshold]

        return [(score, doc_idx) for doc_idx, score in ranked[:k]]


class StructuredChunkedIndex(ChunkedIndex):
    """Section/paragraph chunking with top-n document-score aggregation."""

    def __init__(
        self,
        model_name="sentence-transformers/all-mpnet-base-v2",
        fields=("title", "abstract", "full_text"),
        device=None,
        text_builder=None,
        encoder=None,
        batch_size=64,
        max_words=220,
        min_words=80,
        target_words=180,
        top_n=3,
        aggregation="weighted",
    ):
        super().__init__(
            model_name=model_name,
            fields=fields,
            device=device,
            text_builder=text_builder,
            encoder=encoder,
            batch_size=batch_size,
            max_words=max_words,
            overlap=0,
        )
        self.min_words = min_words
        self.target_words = target_words
        self.top_n = top_n
        self.aggregation = aggregation

    def build(self, dataset, batch_size=None, show_progress_bar=True):
        """Build structure-aware chunks and encode them."""
        self.dataset = dataset
        self.acl_id_to_doc_idx = {}
        self.chunk_records = []
        self.chunk_to_doc = []
        self.doc_to_chunks = {}

        for doc_idx, doc in enumerate(dataset):
            acl_id = str(doc.get("acl_id", ""))
            self.acl_id_to_doc_idx[acl_id] = doc_idx

            chunks = chunk_paper_structured(
                doc,
                max_words=self.max_words,
                min_words=self.min_words,
                target_words=self.target_words,
            )
            if not chunks:
                fallback = str(doc.get("title", "") or "").strip()
                if fallback:
                    chunks = [{"acl_id": acl_id, "chunk_idx": 0, "text": fallback, "source": "fallback"}]

            chunk_indices = []
            for chunk in chunks:
                chunk_indices.append(len(self.chunk_records))
                self.chunk_records.append(chunk)
                self.chunk_to_doc.append(doc_idx)
            self.doc_to_chunks[doc_idx] = chunk_indices

        chunk_texts = [chunk["text"] for chunk in self.chunk_records]
        self.doc_texts = chunk_texts

        embeds = self.encoder.encode(
            chunk_texts,
            batch_size=batch_size or self.batch_size,
            convert_to_numpy=True,
            show_progress_bar=show_progress_bar,
            normalize_embeddings=True,
        )

        self.chunk_embeds = self._sanitize_matrix_embeddings(embeds, "structured chunk embedding")
        self.doc_embeds = self.chunk_embeds
        return self

    def _aggregate_topn(self, sorted_scores):
        if not sorted_scores:
            return -np.inf

        top = sorted_scores[: self.top_n]
        if self.aggregation == "mean":
            return float(np.mean(top))

        # Weighted top-n: s1 + 0.5*s2 + 0.25*s3 ... (normalized by used weights)
        weights = [0.5**i for i in range(len(top))]
        weighted_sum = sum(w * s for w, s in zip(weights, top))
        return float(weighted_sum / sum(weights))

    def retrieve(self, query, k=10, threshold=None):
        """Return top-k (score, doc_idx) using top-n chunk aggregation."""
        if self.chunk_embeds is None:
            raise RuntimeError("Chunked index not built. Call build(dataset) first.")

        query_vec = self.encode_query(query)
        if not np.any(query_vec):
            return []

        chunk_scores = self._score_chunks(query_vec)
        doc_scores = {}

        for chunk_idx, score in enumerate(chunk_scores):
            doc_idx = self.chunk_to_doc[chunk_idx]
            doc_scores.setdefault(doc_idx, []).append(float(score))

        aggregated = []
        for doc_idx, scores in doc_scores.items():
            scores.sort(reverse=True)
            agg = self._aggregate_topn(scores)
            aggregated.append((doc_idx, agg))

        aggregated.sort(key=lambda item: item[1], reverse=True)
        if threshold is not None:
            aggregated = [(doc_idx, score) for doc_idx, score in aggregated if score >= threshold]

        return [(score, doc_idx) for doc_idx, score in aggregated[:k]]


def top_chunks_for_prompt(index, query, paper_ids, k_chunks=3):
    """
    Return query-relevant chunk text per paper for LM selective stuffing.

    Output shape:
      {
        "acl_id": "[chunk i | score s] ...\n[chunk j | score t] ...",
        ...
      }
    """
    if not isinstance(index, ChunkedIndex):
        raise TypeError("top_chunks_for_prompt expects a ChunkedIndex")
    if index.chunk_embeds is None:
        raise RuntimeError("Chunked index not built. Call build(dataset) first.")

    query_vec = index.encode_query(query)
    if not np.any(query_vec):
        return {}

    chunk_scores = index._score_chunks(query_vec)
    result = {}

    for acl_id in paper_ids:
        acl_id = str(acl_id)
        doc_idx = index.acl_id_to_doc_idx.get(acl_id)
        if doc_idx is None:
            continue

        chunk_indices = index.doc_to_chunks.get(doc_idx, [])
        if not chunk_indices:
            continue

        ranked_chunk_indices = sorted(
            chunk_indices,
            key=lambda ci: chunk_scores[ci],
            reverse=True,
        )[:k_chunks]

        lines = []
        for ci in ranked_chunk_indices:
            meta = index.chunk_records[ci]
            lines.append(
                f"[chunk {meta['chunk_idx']} | score {float(chunk_scores[ci]):.3f}] {meta['text']}"
            )

        result[acl_id] = "\n".join(lines)

    return result
