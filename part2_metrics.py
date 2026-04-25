"""Part 2 retrieval evaluation helpers (ACL anthology)."""

from metrics import average_precision, precision_recall_f1


def _as_query_list(gold_queries):
    """Accept either {'queries': [...]} or a raw list of query dicts."""
    if isinstance(gold_queries, dict):
        return gold_queries.get("queries", [])
    return gold_queries


def _answerable_only(queries):
    """Keep only queries with at least one relevant ACL id."""
    return [entry for entry in queries if len(entry.get("r", [])) > 0]


def _safe_div(num, den):
    return num / den if den else 0.0


def _empty_scores():
    return {
        "macro_P": 0.0,
        "macro_R": 0.0,
        "macro_F1": 0.0,
        "micro_P": 0.0,
        "micro_R": 0.0,
        "micro_F1": 0.0,
        "MAP": 0.0,
    }


def _extract_acl_id(item):
    """
    Normalize one ranked item to an ACL id string.

    Supported shapes:
      - "P16-1162"
      - (score, "P16-1162")
      - ("P16-1162", score)
      - {"acl_id": "P16-1162"}
      - {"id": "P16-1162"}
    """
    if isinstance(item, str):
        return item

    if isinstance(item, dict):
        if isinstance(item.get("acl_id"), str):
            return item["acl_id"]
        if isinstance(item.get("id"), str):
            return item["id"]
        return None

    if isinstance(item, (tuple, list)) and len(item) >= 2:
        left, right = item[0], item[1]
        if isinstance(left, str):
            return left
        if isinstance(right, str):
            return right
        return None

    return None


def _compute_from_ranked_ids(ranked_ids_per_query, queries, k=10):
    """Compute macro/micro P,R,F1 and MAP from ranked ACL-id lists."""
    if not queries:
        return _empty_scores()

    precisions, recalls, f1s, aps = [], [], [], []
    total_tp, total_retrieved, total_relevant = 0, 0, 0

    for ranked_ids, entry in zip(ranked_ids_per_query, queries):
        retrieved_ids = list(ranked_ids or [])[:k]
        relevant_ids = set(entry.get("r", []))

        p, r, f1 = precision_recall_f1(retrieved_ids, relevant_ids)
        ap = average_precision(retrieved_ids, relevant_ids)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)
        aps.append(ap)

        tp = len(set(retrieved_ids) & relevant_ids)
        total_tp += tp
        total_retrieved += len(retrieved_ids)
        total_relevant += len(relevant_ids)

    n = len(queries)
    macro_p = sum(precisions) / n
    macro_r = sum(recalls) / n
    macro_f1 = sum(f1s) / n

    micro_p = _safe_div(total_tp, total_retrieved)
    micro_r = _safe_div(total_tp, total_relevant)
    micro_f1 = _safe_div(2 * micro_p * micro_r, micro_p + micro_r)

    return {
        "macro_P": macro_p,
        "macro_R": macro_r,
        "macro_F1": macro_f1,
        "micro_P": micro_p,
        "micro_R": micro_r,
        "micro_F1": micro_f1,
        "MAP": sum(aps) / n,
    }


def evaluate_neural(index, gold_queries, k=10):
    """
    Evaluate a neural retriever with Part 2's ACL-id query format.

    Expected index interface:
      - index.retrieve(query: str, k: int) -> list[(score, doc_idx)]
      - index.acl_ids(indices: list[int]) -> list[str]
    """
    queries = _answerable_only(_as_query_list(gold_queries))
    if not queries:
        return _empty_scores()

    ranked_ids_per_query = []
    for entry in queries:
        ranked = index.retrieve(entry["q"], k=k)
        indices = [idx for _, idx in ranked]
        retrieved_ids = index.acl_ids(indices)
        ranked_ids_per_query.append(retrieved_ids)

    return _compute_from_ranked_ids(ranked_ids_per_query, queries, k=k)


def evaluate_ranked_lists(ranked_lists, gold_queries, k=10):
    """
    Evaluate precomputed ranked ACL-id lists.

    ranked_lists can be either:
      - list aligned with answerable gold queries
      - dict keyed by query index or query text
    Each ranked list can contain ACL ids directly or tuples/dicts that include
    ACL ids (see _extract_acl_id).
    """
    queries = _answerable_only(_as_query_list(gold_queries))
    if not queries:
        return _empty_scores()

    resolved_lists = []

    if isinstance(ranked_lists, dict):
        for idx, entry in enumerate(queries):
            raw = ranked_lists.get(idx, ranked_lists.get(entry.get("q"), []))
            ids = []
            for item in raw:
                acl_id = _extract_acl_id(item)
                if acl_id is not None:
                    ids.append(acl_id)
            resolved_lists.append(ids)
    else:
        for raw in list(ranked_lists)[: len(queries)]:
            ids = []
            for item in raw:
                acl_id = _extract_acl_id(item)
                if acl_id is not None:
                    ids.append(acl_id)
            resolved_lists.append(ids)

        while len(resolved_lists) < len(queries):
            resolved_lists.append([])

    return _compute_from_ranked_ids(resolved_lists, queries, k=k)


def compare_table(results_by_name):
    """
    Print a compact markdown-style comparison table and return it as a string.

    Expected input:
      {
        "baseline": {"macro_P": ..., "macro_R": ..., ... "MAP": ...},
        "variant":  {...},
      }
    """
    columns = [
        "macro_P",
        "macro_R",
        "macro_F1",
        "micro_P",
        "micro_R",
        "micro_F1",
        "MAP",
    ]

    lines = [
        "| System | macro_P | macro_R | macro_F1 | micro_P | micro_R | micro_F1 | MAP |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for name, metrics_dict in results_by_name.items():
        row = [name]
        for key in columns:
            value = metrics_dict.get(key)
            row.append(f"{value:.4f}" if isinstance(value, (int, float)) else "-")
        lines.append("| " + " | ".join(row) + " |")

    table = "\n".join(lines)
    print(table)
    return table
