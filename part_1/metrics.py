# metrics.py
# Shared evaluation functions imported by evaluate.py and field_experiment.py

from tf_idf import vectorize_query, dot_sparse

def retrieve_ids(query, doc_vectors, vocab, idf, bigram, k=10, threshold=0):
    """Return a ranked list of (score, doc_idx) for a single query string."""
    query_vec = vectorize_query(query, bigram, vocab, idf)
    if not query_vec:
        return []
    scores = [(dot_sparse(query_vec, vec), idx) for idx, vec in enumerate(doc_vectors)]
    scores.sort(reverse=True)
    return [(score, idx) for score, idx in scores if score >= threshold][:k]

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
    hits, total  = 0, 0.0
    for k, doc_id in enumerate(ranked_ids, start=1):
        if doc_id in relevant_set:
            hits  += 1
            total += hits / k
    return total / len(relevant_set)

def precision_recall_f1(retrieved_ids, relevant_ids):
    """Compute P, R, F1 for a single query."""
    if not retrieved_ids:
        return 0.0, 0.0, 0.0
    tp = len(set(retrieved_ids) & set(relevant_ids))
    p  = tp / len(retrieved_ids)
    r  = tp / len(relevant_ids) if relevant_ids else 0.0
    f1 = (2 * p * r / (p + r)) if (p + r) > 0 else 0.0
    return p, r, f1

def evaluate(doc_vectors, bigram, vocab, idf, gold_queries, k=10):
    """Return macro P, R, F1, MAP for a given set of doc vectors."""
    precisions, recalls, f1s, aps = [], [], [], []
    for entry in gold_queries:
        relevant_ids = set(doc_id for doc_id, _ in entry["r"])
        query_vec    = vectorize_query(entry["q"], bigram, vocab, idf)
        ranked       = retrieve_ids(query_vec, doc_vectors, k)

        tp = len(set(ranked) & relevant_ids)
        p  = tp / len(ranked)       if ranked       else 0.0
        r  = tp / len(relevant_ids) if relevant_ids else 0.0
        f1 = (2 * p * r / (p + r)) if (p + r) > 0  else 0.0
        ap = average_precision(ranked, relevant_ids)

        precisions.append(p); recalls.append(r); f1s.append(f1); aps.append(ap)

    n = len(gold_queries)
    return {
        "macro_P"  : sum(precisions) / n,
        "macro_R"  : sum(recalls)    / n,
        "macro_F1" : sum(f1s)        / n,
        "MAP"      : sum(aps)        / n,
    }