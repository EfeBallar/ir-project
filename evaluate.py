# Evaluation of the TF-IDF retrieval system against gold queries.
# Run: python evaluate.py
# Estimated runtime: 5 min

import json, datasets
from tf_idf import doc_to_text, tokenize_docs, train_bigram_model, vectorize, build_vocab
from metrics import retrieve_ids, average_precision, precision_recall_f1

# ── Data loading ────────────────────────────────────────────────────────────
DATA_PATH = "./irse_documents_2026_recipes.parquet"
dataset = datasets.load_dataset("parquet", data_files=DATA_PATH)["train"]
gold_queries = json.load(open("./irse_queries_2026_recipes.json", "r"))

# ── Indexing ─────────────────────────────────────────────────────────────────
# 'steps' field is excluded from tokenization - check 'tf-idf.py' file for fields.
doc_texts = [doc_to_text(doc) for doc in dataset]
tokenized_docs = tokenize_docs(doc_texts)
bigram = train_bigram_model(tokenized_docs)
tokenized_docs = [bigram[doc] for doc in tokenized_docs]
vocab, idf, df = build_vocab(tokenized_docs, min_df=5, max_df_ratio=0.5, max_vocab=50000)
print("Built vocab.")
doc_vectors = [vectorize(tokens, vocab, idf) for tokens in tokenized_docs]
print("Vectorized docs.")

# Filter out unanswerable queries (empty r) — they have no relevant docs
answerable_queries = [q for q in gold_queries["queries"] if len(q["r"]) > 0]
print(f"Loaded {len(gold_queries['queries'])} queries, {len(answerable_queries)} answerable.")

# ── Main evaluation (k=10, no threshold) ─────────────────────────────────────
precisions, recalls, f1s, aps = [], [], [], []

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
# test below which scores of retrieved docs are considered irrelevant
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
# test number of retrieved docs
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

"""
Loaded 47 queries, 41 answerable.

── Evaluation Results ──
  Macro  Precision : 0.0805
  Macro  Recall    : 0.1148
  Macro  F1        : 0.0798
  Micro  Precision : 0.0805
  Micro  Recall    : 0.0570
  Micro  F1        : 0.0667
  MAP              : 0.0618
──────────────────────────

── Threshold sensitivity (macro F1) ────────────────────────
  threshold=0.05  macro-F1=0.0953  macro-P=0.1626  macro-R=0.0826
  threshold=0.10  macro-F1=0.0953  macro-P=0.1626  macro-R=0.0826
  threshold=0.15  macro-F1=0.0953  macro-P=0.1626  macro-R=0.0826
  threshold=0.20  macro-F1=0.0953  macro-P=0.1626  macro-R=0.0826
  threshold=0.25  macro-F1=0.0883  macro-P=0.1545  macro-R=0.0765
  threshold=0.30  macro-F1=0.0886  macro-P=0.1667  macro-R=0.0765
  threshold=0.35  macro-F1=0.0729  macro-P=0.1423  macro-R=0.0645
  threshold=0.40  macro-F1=0.0459  macro-P=0.0732  macro-R=0.0376
────────────────────────────────────────────────────────────

── k sensitivity (macro F1) ────────────────────────
  k=3  macro-F1=0.0953  macro-P=0.1626  macro-R=0.0826
  k=5  macro-F1=0.0774  macro-P=0.1024  macro-R=0.0832
  k=10  macro-F1=0.0798  macro-P=0.0805  macro-R=0.1148
  k=20  macro-F1=0.0688  macro-P=0.0549  macro-R=0.1476
  k=50  macro-F1=0.0475  macro-P=0.0307  macro-R=0.1778
──────────────────────────
"""