# field_experiment.py
# Standalone experiment: which field combination yields the best retrieval?
# Run: python field_experiment.py
# Estimated runtime: ~10 min

import json, datasets
from gensim.models import Phrases
from gensim.models.phrases import Phraser

from metrics import average_precision, retrieve_ids
from tf_idf import build_vocab, preprocess, vectorize

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH = "./irse_documents_2026_recipes.parquet"
QUERIES_PATH = "./irse_queries_2026_recipes.json"
K = 10

FIELD_COMBINATIONS = [
    ["name"],
    ["name", "tags"],
    ["name", "ingredients"],
    ["name", "ingredients", "tags"],
    ["name", "ingredients", "description"],
    ["name", "ingredients", "description", "tags"],
    ["name", "ingredients", "description", "tags", "steps"],  # everything
]

# ── doc_to_text with variable fields (unique to this experiment) ──────────────
def doc_to_text(doc, fields):
    parts = []
    for field in fields:
        val = doc.get(field, "") or ""
        if isinstance(val, list):
            val = " ".join(val)
        parts.append(val)
    return " ".join(parts)

# ── Evaluation  ────────────────────────────────────────────────────────
def evaluate(doc_vectors, bigram, vocab, idf, gold_queries, k=K):
    precisions, recalls, f1s, aps = [], [], [], []
    for entry in gold_queries:
        relevant_ids = set(doc_id for doc_id, _ in entry["r"])
        ranked = retrieve_ids(entry["q"], doc_vectors, vocab, idf, bigram, k=k)
        retrieved_ids = [idx for _, idx in ranked]  # ← unpack (score, idx) tuples

        tp = len(set(retrieved_ids) & relevant_ids)
        p = tp / len(retrieved_ids) if retrieved_ids else 0.0
        r = tp / len(relevant_ids) if relevant_ids else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
        ap = average_precision(retrieved_ids, relevant_ids)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        aps.append(ap)

    n = len(gold_queries)
    return {
        "macro_P": sum(precisions) / n,
        "macro_R": sum(recalls) / n,
        "macro_F1": sum(f1s) / n,
        "MAP": sum(aps) / n,
    }

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading dataset and queries...")
    dataset = datasets.load_dataset("parquet", data_files=DATA_PATH)["train"]
    gold_queries = json.load(open(QUERIES_PATH))["queries"]
    gold_queries = [q for q in gold_queries if len(q["r"]) > 0]

    # Train bigram model on all fields
    print("Training bigram model...")
    base_tokens = [
        preprocess(doc_to_text(doc, ["name", "ingredients", "description", "tags", "steps"]))
        for doc in dataset
    ]
    phrases = Phrases(base_tokens, min_count=5, threshold=3.0)
    bigram = Phraser(phrases)

    results = []

    for fields in FIELD_COMBINATIONS:
        label = " + ".join(fields)

        tokenized_docs = [preprocess(doc_to_text(doc, fields)) for doc in dataset]
        tokenized_docs = [bigram[doc] for doc in tokenized_docs]
        vocab, idf, _ = build_vocab(tokenized_docs)
        doc_vectors = [vectorize(tokens, vocab, idf) for tokens in tokenized_docs]

        metrics = evaluate(doc_vectors, bigram, vocab, idf, gold_queries)
        metrics["fields"] = label
        results.append(metrics)

    print(f"{'Fields':<55} {'macro-P':>8} {'macro-R':>8} {'macro-F1':>9} {'MAP':>8}")
    print("─" * 90)
    best = max(results, key=lambda x: x["MAP"])
    for r in results:
        marker = " ← best MAP" if r["fields"] == best["fields"] else ""
        print(
            f"{r['fields']:<55} {r['macro_P']:>8.4f} {r['macro_R']:>8.4f} "
            f"{r['macro_F1']:>9.4f} {r['MAP']:>8.4f}{marker}"
        )
    print("-" * 90)
    print(f"Best field combination: [{best['fields']}]  MAP={best['MAP']:.4f}")

"""
Fields                                                   macro-P  macro-R  macro-F1      MAP
──────────────────────────────────────────────────────────────────────────────────────────
name                                                      0.0537   0.0707    0.0536   0.0383
name + tags                                               0.0463   0.0694    0.0483   0.0424
name + ingredients                                        0.0463   0.0704    0.0496   0.0298
name + ingredients + tags                                 0.0585   0.0763    0.0573   0.0368
name + ingredients + description                          0.0756   0.0883    0.0699   0.0374
name + ingredients + description + tags                   0.0805   0.0960    0.0753   0.0445 ← best MAP
name + ingredients + description + tags + steps           0.0756   0.0868    0.0701   0.0334
──────────────────────────────────────────────────────────────────────────────────────────
Best field combination: [name + ingredients + description + tags]  MAP=0.0445
"""