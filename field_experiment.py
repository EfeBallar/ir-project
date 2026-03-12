# field_experiment.py
# Standalone experiment: which field combination yields the best retrieval?
# Run: python field_experiment.py
# Estimated runtime: ~10-20 min depending on machine (vectorization dominates)

import json
import math
import re
from collections import Counter

import datasets
import numpy as np
from gensim.models import Phrases
from gensim.models.phrases import Phraser
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH    = "./irse_documents_2026_recipes.parquet"
QUERIES_PATH = "./irse_queries_2026_recipes.json"
K            = 10          # fixed k (optimal from sensitivity experiment)
MIN_DF       = 5
MAX_DF_RATIO = 0.5
MAX_VOCAB    = 50000

FIELD_COMBINATIONS = [
    ["name"],
    ["name", "tags"],
    ["name", "ingredients"],
    ["name", "ingredients", "tags"],
    ["name", "ingredients", "description"],
    ["name", "ingredients", "description", "tags"],
    ["name", "ingredients", "description", "tags", "steps"],  # everything
]

# ── Preprocessing ─────────────────────────────────────────────────────────────
STOPWORDS  = set(stopwords.words("english"))
DIGITS     = re.compile(r"\d")
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    words = re.findall(r"\b[a-z]+\b", text.lower())
    words = [w for w in words if w not in STOPWORDS]
    words = [w for w in words if not DIGITS.search(w)]
    words = [lemmatizer.lemmatize(w) for w in words]
    return words

def doc_to_text(doc, fields):
    """Concatenate the requested fields of a recipe into one string."""
    parts = []
    for field in fields:
        val = doc.get(field, "") or ""
        if isinstance(val, list):
            val = " ".join(val)
        parts.append(val)
    return " ".join(parts)

# ── TF-IDF core ───────────────────────────────────────────────────────────────
def build_vocab(tokenized_docs, min_df=MIN_DF, max_df_ratio=MAX_DF_RATIO, max_vocab=MAX_VOCAB):
    df = Counter()
    for tokens in tokenized_docs:
        for term in set(tokens):
            df[term] += 1
    doc_count = len(tokenized_docs)
    max_df    = int(max_df_ratio * doc_count)
    terms     = [t for t, d in df.items() if min_df <= d <= max_df]
    terms.sort(key=lambda t: (-df[t], t))
    if len(terms) > max_vocab:
        terms = terms[:max_vocab]
    vocab = {term: idx for idx, term in enumerate(terms)}
    idf   = np.zeros(len(vocab), dtype=np.float32)
    for term, idx in vocab.items():
        idf[idx] = math.log((doc_count + 1) / (df[term] + 1)) + 1.0
    return vocab, idf

def vectorize(tokens, vocab, idf):
    counts = Counter(tokens)
    vec    = {}
    for term, count in counts.items():
        idx = vocab.get(term)
        if idx is None:
            continue
        tf        = 1.0 + math.log(count)
        vec[idx]  = tf * float(idf[idx])
    norm = math.sqrt(sum(v * v for v in vec.values()))
    if norm > 0:
        for key in vec:
            vec[key] /= norm
    return vec

def dot_sparse(a, b):
    if len(a) > len(b):
        a, b = b, a
    return sum(v * b[k] for k, v in a.items() if k in b)

def vectorize_query(query, bigram, vocab, idf):
    tokens = preprocess(query)
    tokens = bigram[tokens]
    return vectorize(tokens, vocab, idf)

# ── Evaluation helpers ────────────────────────────────────────────────────────
def retrieve_ids(query_vec, doc_vectors, k):
    """Return ranked list of doc indices for a pre-computed query vector."""
    if not query_vec:
        return []
    scores = [(dot_sparse(query_vec, vec), idx) for idx, vec in enumerate(doc_vectors)]
    scores.sort(reverse=True)
    return [idx for _, idx in scores[:k]]

def average_precision(ranked_ids, relevant_ids):
    if not relevant_ids:
        return 0.0
    relevant_set = set(relevant_ids)
    hits, total  = 0, 0.0
    for k, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_set:
            hits  += 1
            total += hits / k
    return total / len(relevant_set)

def evaluate(doc_vectors, bigram, vocab, idf, gold_queries, k=K):
    """Return macro P, R, F1, MAP for a given set of doc vectors."""
    precisions, recalls, f1s, aps = [], [], [], []
    for entry in gold_queries:
        query        = entry["q"]
        relevant_ids = set(doc_id for doc_id, _ in entry["r"])
        query_vec    = vectorize_query(query, bigram, vocab, idf)
        ranked       = retrieve_ids(query_vec, doc_vectors, k)

        tp = len(set(ranked) & relevant_ids)
        p  = tp / len(ranked)       if ranked        else 0.0
        r  = tp / len(relevant_ids) if relevant_ids  else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0   else 0.0
        ap = average_precision(ranked, relevant_ids)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        aps.append(ap)

    n = len(gold_queries)
    return {
        "macro_P"  : sum(precisions) / n,
        "macro_R"  : sum(recalls)    / n,
        "macro_F1" : sum(f1s)        / n,
        "MAP"      : sum(aps)        / n,
    }

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading dataset and queries...")
    dataset      = datasets.load_dataset("parquet", data_files=DATA_PATH)["train"]
    gold_queries = json.load(open(QUERIES_PATH))["queries"]

    # Train bigram model once on name+ingredients (lightweight base)
    print("Training bigram model...")
    base_texts   = [doc_to_text(doc, ["name", "ingredients"]) for doc in dataset]
    base_tokens  = [preprocess(t) for t in base_texts]
    phrases      = Phrases(base_tokens, min_count=5, threshold=3.0)
    bigram       = Phraser(phrases)

    results = []

    for fields in FIELD_COMBINATIONS:
        label = " + ".join(fields)
        print(f"── Running experiment: [{label}]")

        # Tokenize with current field combination
        doc_texts      = [doc_to_text(doc, fields) for doc in dataset]
        tokenized_docs = [preprocess(t) for t in doc_texts]
        tokenized_docs = [bigram[doc] for doc in tokenized_docs]

        vocab, idf  = build_vocab(tokenized_docs)
        doc_vectors = [vectorize(tokens, vocab, idf) for tokens in tokenized_docs]

        metrics = evaluate(doc_vectors, bigram, vocab, idf, gold_queries)
        metrics["fields"] = label
        results.append(metrics)

        print(f"   macro-P={metrics['macro_P']:.4f}  "
              f"macro-R={metrics['macro_R']:.4f}  "
              f"macro-F1={metrics['macro_F1']:.4f}  "
              f"MAP={metrics['MAP']:.4f}\n")

    # ── Summary table ─────────────────────────────────────────────────────────
    print("══ Field Selection Summary ══════════════════════════════════════")
    print(f"{'Fields':<55} {'macro-P':>8} {'macro-R':>8} {'macro-F1':>9} {'MAP':>8}")
    print("─" * 90)
    best = max(results, key=lambda x: x["MAP"])
    for r in results:
        marker = " ← best MAP" if r["fields"] == best["fields"] else ""
        print(f"{r['fields']:<55} {r['macro_P']:>8.4f} {r['macro_R']:>8.4f} "
              f"{r['macro_F1']:>9.4f} {r['MAP']:>8.4f}{marker}")
    print("═" * 90)
    print(f"\nBest field combination: [{best['fields']}]  MAP={best['MAP']:.4f}")