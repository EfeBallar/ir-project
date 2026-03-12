# source venv/bin/activate
import json, datasets
from tf_idf import doc_to_text, tokenize_docs, train_bigram_model, vectorize, vectorize_query, build_vocab, dot_sparse

# ── Data loading ────────────────────────────────────────────────────────────
DATA_PATH = "./irse_documents_2026_recipes.parquet"
dataset = datasets.load_dataset("parquet", data_files=DATA_PATH)["train"]
gold_queries = json.load(open("./irse_queries_2026_recipes.json", "r"))
retrieval_queries = [
    "cajun style gumbo with an easy roux",
    "shrimp tacos recipe for tonight",
    "vegetarian lasagna but easy",
    "spageti and meatballs?",
    "lunch baked at 200 °C",
]

# ── Indexing ─────────────────────────────────────────────────────────────────
doc_texts = [doc_to_text(doc) for doc in dataset]
tokenized_docs = tokenize_docs(doc_texts)
bigram = train_bigram_model(tokenized_docs)
tokenized_docs = [bigram[doc] for doc in tokenized_docs]
vocab, idf, df = build_vocab(tokenized_docs, min_df=5, max_df_ratio=0.5, max_vocab=50000)
print("Built vocab")
doc_vectors = [vectorize(tokens, vocab, idf) for tokens in tokenized_docs]
print("Vectorized docs")

# ── Nearest-neighbour retrieval  ─────────────────────────────
# ── Evaluation helpers ───────────────────────────────────────────────────────

def retrieve_ids(query, doc_vectors, vocab, idf, bigram, k=3, threshold=0.3):
    """Return a ranked list of (score, doc_idx) for a single query string."""
    query_vec = vectorize_query(query, bigram, vocab, idf)
    if not query_vec:
        return []
    scores = [(dot_sparse(query_vec, vec), idx) for idx, vec in enumerate(doc_vectors)]
    scores.sort(reverse=True)
    return [(score, idx) for score, idx in scores if score >= threshold][:k]

def precision_recall_f1(retrieved_ids, relevant_ids):
    """Compute P, R, F1 for a single query."""
    if not retrieved_ids:
        return 0.0, 0.0, 0.0
    tp = len(set(retrieved_ids) & set(relevant_ids))
    p  = tp / len(retrieved_ids)
    r  = tp / len(relevant_ids) if relevant_ids else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1

def average_precision(ranked_ids, relevant_ids):
    """
    Compute Average Precision (AP) for a single query.

    AP = (1 / |R|) * sum_{k=1}^{K} P@k * rel(k)

    where:
      |R|    = total number of relevant documents for the query
      K      = total number of retrieved documents
      P@k    = precision computed on the top-k retrieved documents
      rel(k) = 1 if the document at rank k is relevant, 0 otherwise

    'At k' means: considering only the first k documents in the ranked list.
    """
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    hits, total = 0, 0.0
    for k, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_set:
            hits += 1
            total += hits / k   # P@k * rel(k)
    return total / len(relevant_set)

# ── Run evaluation over gold queries ────────────────────────────────────────
precisions, recalls, f1s, aps = [], [], [], []
# Filter out unanswerable queries (empty r) — they have no relevant docs
answerable_queries = [q for q in gold_queries["queries"] if len(q["r"]) > 0]
for entry in answerable_queries:
    query        = entry["q"]
    relevant_ids = set(doc_id for doc_id, _ in entry["r"])

    ranked = retrieve_ids(query, doc_vectors, vocab, idf, bigram, k=10, threshold=0)
    retrieved_ids = [idx for _, idx in ranked]   # ordered list for AP

    p, r, f1 = precision_recall_f1(retrieved_ids, relevant_ids)
    ap        = average_precision(retrieved_ids, relevant_ids)

    precisions.append(p)
    recalls.append(r)
    f1s.append(f1)
    aps.append(ap)

# ── Macro averages (mean over queries) ──────────────────────────────────────
macro_p  = sum(precisions) / len(precisions)
macro_r  = sum(recalls)    / len(recalls)
macro_f1 = sum(f1s)        / len(f1s)

# ── Micro averages (aggregate TP/FP/FN first, then compute) ─────────────────
total_tp, total_retrieved, total_relevant = 0, 0, 0

for entry in answerable_queries:
    query        = entry["q"]
    relevant_ids = set(doc_id for doc_id, _ in entry["r"])
    ranked       = retrieve_ids(query, doc_vectors, vocab, idf, bigram, k=10, threshold=0)
    retrieved_ids = [idx for _, idx in ranked]

    tp = len(set(retrieved_ids) & relevant_ids)
    total_tp        += tp
    total_retrieved += len(retrieved_ids)
    total_relevant  += len(relevant_ids)

micro_p  = total_tp / total_retrieved if total_retrieved > 0 else 0.0
micro_r  = total_tp / total_relevant  if total_relevant  > 0 else 0.0
micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0

# ── MAP ──────────────────────────────────────────────────────────────────────
# MAP = (1 / |Q|) * sum_{j=1}^{|Q|} AP(q_j)
# where |Q| = number of queries, AP(q_j) = average precision for query j
MAP = sum(aps) / len(aps)

# ── Report ───────────────────────────────────────────────────────────────────
print("\n── Evaluation Results ──")
print(f"  Macro  Precision : {macro_p:.4f}")
print(f"  Macro  Recall    : {macro_r:.4f}")
print(f"  Macro  F1        : {macro_f1:.4f}")
print(f"  Micro  Precision : {micro_p:.4f}")
print(f"  Micro  Recall    : {micro_r:.4f}")
print(f"  Micro  F1        : {micro_f1:.4f}")
print(f"  MAP              : {MAP:.4f}")
print("──────────────────────────\n")

# ── Threshold sensitivity analysis  ───────────────
print("── Threshold sensitivity (macro F1) ────────────────────────")
for t in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40]:
    ps, rs, fs = [], [], []
    for entry in answerable_queries:
        rel = set(doc_id for doc_id, _ in entry["r"])
        ranked = retrieve_ids(entry["q"], doc_vectors, vocab, idf, bigram, k=3, threshold=t)
        ret = [idx for _, idx in ranked]
        p, r, f1 = precision_recall_f1(ret, rel)
        ps.append(p); rs.append(r); fs.append(f1)
    print(f"  threshold={t:.2f}  macro-F1={sum(fs)/len(fs):.4f}  "
          f"macro-P={sum(ps)/len(ps):.4f}  macro-R={sum(rs)/len(rs):.4f}")
print("────────────────────────────────────────────────────────────\n")

# ── k sensitivity analysis  ───────────────
print("── k sensitivity (macro F1) ────────────────────────")
for k in [3, 5, 10, 20, 50]:
    ps, rs, fs = [], [], []
    for entry in answerable_queries:
        rel = set(doc_id for doc_id, _ in entry["r"])
        ranked = retrieve_ids(entry["q"], doc_vectors, vocab, idf, bigram, k=k, threshold=0)
        ret = [idx for _, idx in ranked]
        p, r, f1 = precision_recall_f1(ret, rel)
        ps.append(p); rs.append(r); fs.append(f1)
    print(f"  k={k}  macro-F1={sum(fs)/len(fs):.4f}  "
          f"macro-P={sum(ps)/len(ps):.4f}  macro-R={sum(rs)/len(rs):.4f}")
print("──────────────────────────\n")
