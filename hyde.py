"""HyDE helpers for Part 2 retrieval experiments."""

import torch


HYDE_PROMPT = (
    "<s>[INST] You are helping retrieval over NLP research papers. "
    "Write a short academic-style abstract (about 150 words) that would likely answer "
    "the user question below.\n\n"
    "Rules:\n"
    "- Write only the abstract text.\n"
    "- Use terminology common in NLP papers.\n"
    "- Do not include lists, bullet points, or meta-commentary.\n\n"
    "Question: {query}\n\n"
    "Hypothetical abstract:\n"
    "[/INST]"
)


def _extract_assistant_text(decoded_text):
    if "[/INST]" in decoded_text:
        return decoded_text.split("[/INST]")[-1].strip()
    return decoded_text.strip()


def _generate_hypothetical_abstract(query, model=None, tokenizer=None, max_new_tokens=220):
    if model is None or tokenizer is None:
        raise ValueError("HyDE requires both model and tokenizer.")

    prompt = HYDE_PROMPT.format(query=str(query))
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
    return _extract_assistant_text(decoded)


def hyde_retrieve(
    query,
    index,
    model=None,
    tokenizer=None,
    k=10,
    max_new_tokens=220,
    return_hypothesis=False,
):
    """
    Generate a hypothetical abstract for `query`, then retrieve with its embedding.

    Retrieval uses a robust HyDE text: original query + hypothetical abstract.
    Returns ranked ACL ids; optionally also returns the generated abstract.
    """
    hypothetical_abstract = _generate_hypothetical_abstract(
        query,
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
    )

    hyde_query_text = f"{query}\n\n{hypothetical_abstract}".strip()
    ranked = index.retrieve(hyde_query_text, k=k)
    ranked_ids = index.acl_ids([idx for _, idx in ranked])

    if return_hypothesis:
        return ranked_ids, hypothetical_abstract
    return ranked_ids
