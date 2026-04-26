"""Query rewriting helpers for Part 2 retrieval experiments."""

import re

import torch


REWRITE_PROMPT = (
    "<s>[INST] You rewrite user search questions for retrieval over NLP research papers. "
    "Given one question, generate {n} alternative rewritten queries.\n\n"
    "Rules:\n"
    "- Keep exactly the same information need as the original query.\n"
    "- Do not broaden, narrow, or change the target concept.\n"
    "- Do not explain the topic; output search queries only.\n"
    "- Use paraphrase, acronym/terminology expansion, and likely domain synonyms.\n"
    "- Produce one rewrite per line.\n"
    "- Do not use numbering, bullets, or any preamble.\n"
    "- Do not generate teaching questions or generic background questions.\n\n"
    "Original query: {query}\n\n"
    "Rewritten queries:\n"
    "[/INST]"
)


def _extract_assistant_text(decoded_text):
    if "[/INST]" in decoded_text:
        return decoded_text.split("[/INST]")[-1].strip()
    return decoded_text.strip()


def _clean_rewrite_line(line):
    line = line.strip()
    if not line:
        return ""

    line = re.sub(r"^\s*(?:[-*•]|\d+[\)\.:]|\(\d+\))\s*", "", line)
    line = re.sub(r"^(?:rewrite|rewritten query|query|alternative)\s*\d*\s*[:\-]\s*", "", line, flags=re.IGNORECASE)
    line = line.strip().strip("`\"'")
    return line.strip()


def _normalize_for_dedup(text):
    return " ".join(str(text).lower().split())


def rewrite_query(query, n=3, model=None, tokenizer=None, max_new_tokens=180):
    """
    Generate N rewritten queries using the provided LM.

    Returns up to N cleaned rewrite strings.
    """
    if model is None or tokenizer is None:
        raise ValueError("rewrite_query requires both model and tokenizer.")
    if n <= 0:
        return []

    prompt = REWRITE_PROMPT.format(query=str(query), n=int(n))
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)

    target_device = getattr(model, "device", None)
    if target_device is None:
        try:
            target_device = next(model.parameters()).device
        except StopIteration:
            target_device = torch.device("cpu")

    encoded = {key: value.to(target_device) for key, value in encoded.items()}

    with torch.no_grad():
        generated_ids = model.generate(
            **encoded,
            max_new_tokens=max_new_tokens,
            do_sample=False,
        )

    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    raw = _extract_assistant_text(decoded)

    rewrites = []
    seen = {_normalize_for_dedup(query)}
    for line in raw.splitlines():
        cleaned = _clean_rewrite_line(line)
        if not cleaned:
            continue
        norm = _normalize_for_dedup(cleaned)
        if not norm or norm in seen:
            continue
        seen.add(norm)
        rewrites.append(cleaned)

    return rewrites[:n]


def _extract_doc_id(item):
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


def rrf_fuse(ranked_lists, k=10, k_rrf=60):
    """Reciprocal Rank Fusion over multiple ranked lists."""
    scores = {}

    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            doc_id = _extract_doc_id(item)
            if doc_id is None:
                continue
            scores[doc_id] = scores.get(doc_id, 0.0) + (1.0 / (k_rrf + rank))

    fused = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return [doc_id for doc_id, _ in fused[:k]]
