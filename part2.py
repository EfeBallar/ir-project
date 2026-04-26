import argparse

from part2_data import build_sample_if_missing, load_anthology, load_queries, answerable_queries
from neural_retrieval import NeuralIndex
from long_docs import ChunkedIndex, StructuredChunkedIndex
from part2_metrics import evaluate_neural, evaluate_ranked_lists, compare_table
from query_rewriting import rewrite_query, rrf_fuse
from hyde import hyde_retrieve
from tf_idf import tokenize_docs, train_bigram_model, build_vocab, vectorize, vectorize_query, dot_sparse

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
EVAL_K = 10
FULL_TEXT_TRUNC_WORDS = 220


def _banner(label, eta):
    print(f"\n[Part 2] {label}  (~{eta})")


def _truncate_words(text, max_words=FULL_TEXT_TRUNC_WORDS):
    words = str(text or "").split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def _title_abstract_fulltext_truncated(doc, max_words=FULL_TEXT_TRUNC_WORDS):
    parts = []

    title = str(doc.get("title", "") or "").strip()
    if title:
        parts.append(title)

    abstract = str(doc.get("abstract", "") or "").strip()
    if abstract:
        parts.append(abstract)

    full_text = _truncate_words(doc.get("full_text", ""), max_words=max_words)
    if full_text:
        parts.append(full_text)

    return "\n".join(parts)


def _paper_title_abstract_text(doc):
    title = str(doc.get("title", "") or "").strip()
    abstract = str(doc.get("abstract", "") or "").strip()
    return f"{title} {abstract}".strip()


def run_tfidf_acl_baseline(anthology, queries, k=EVAL_K):
    """Part 2 lexical baseline on ACL papers using title+abstract fields."""
    print("[Part 2][1/6] TF-IDF baseline on ACL (title+abstract, k=10) (~1 min CPU)")

    doc_texts = [_paper_title_abstract_text(doc) for doc in anthology]
    tokenized_docs = tokenize_docs(doc_texts)
    bigram = train_bigram_model(tokenized_docs)
    tokenized_docs = [bigram[tokens] for tokens in tokenized_docs]
    vocab, idf, _ = build_vocab(tokenized_docs, min_df=5, max_df_ratio=0.5, max_vocab=50000)
    doc_vectors = [vectorize(tokens, vocab, idf) for tokens in tokenized_docs]

    ranked_acl_lists = []
    for entry in queries:
        query_vec = vectorize_query(entry["q"], bigram, vocab, idf)
        if not query_vec:
            ranked_acl_lists.append([])
            continue

        scores = [(dot_sparse(query_vec, vec), idx) for idx, vec in enumerate(doc_vectors)]
        scores.sort(reverse=True)
        top_indices = [idx for _, idx in scores[:k]]
        ranked_acl_lists.append([str(anthology[idx].get("acl_id", "")) for idx in top_indices])

    metrics = evaluate_ranked_lists(ranked_acl_lists, queries, k=k)
    print(
        "[TF-IDF] title+abstract: "
        f"MAP@10={metrics['MAP@10']:.4f} "
        f"macro_F1={metrics['macro_F1']:.4f} "
        f"micro_F1={metrics['micro_F1']:.4f}"
    )
    return metrics


def _print_subword_demo(index):
    tokenizer = getattr(index.encoder, "tokenizer", None)
    if tokenizer is None:
        print("[Neural] Tokenizer demo unavailable (encoder has no tokenizer attribute).")
        return

    word = "Morfessor"
    pieces = tokenizer.tokenize(word)
    print(f"[Neural] Subword demo for '{word}': {pieces}")


def run_neural_baseline(
    anthology,
    queries,
    model_name=MODEL_NAME,
    device=None,
    k=EVAL_K,
):
    """
    Part 2 step 1: neural baseline + field experiment.

    Field variants:
      1) title only
      2) title + abstract (baseline)
      3) title + abstract + truncated full_text
    """
    print("[Part 2][1/6] neural baseline + field experiment + compare against TF-IDF (~4 min on T4)")
    print(f"[Neural] Docs: {len(anthology)} | Queries: {len(queries)} | k={k}")

    tfidf_metrics = run_tfidf_acl_baseline(anthology, queries, k=k)

    # ~2 min on T4
    baseline_index = NeuralIndex(
        model_name=model_name,
        fields=("title", "abstract"),
        device=device,
    )
    baseline_index.build(anthology)
    baseline_metrics = evaluate_neural(baseline_index, queries, k=k)
    print(
        "[Neural] Baseline (title+abstract): "
        f"MAP@10={baseline_metrics['MAP@10']:.4f} "
        f"macro_F1={baseline_metrics['macro_F1']:.4f} "
        f"micro_F1={baseline_metrics['micro_F1']:.4f}"
    )
    print("[Part 2] Retrieval comparison on ACL (as requested in Q1):")
    compare_table(
        {
            "TF-IDF (title+abstract)": tfidf_metrics,
            "Neural (title+abstract)": baseline_metrics,
        }
    )

    # Reuse loaded encoder to avoid re-downloading/re-loading weights.
    shared_encoder = baseline_index.encoder

    title_only_index = NeuralIndex(
        model_name=model_name,
        fields=("title",),
        encoder=shared_encoder,
    )
    title_only_index.build(anthology)
    title_only_metrics = evaluate_neural(title_only_index, queries, k=k)

    title_abs_full_index = NeuralIndex(
        model_name=model_name,
        fields=("title", "abstract"),
        encoder=shared_encoder,
        text_builder=_title_abstract_fulltext_truncated,
    )
    title_abs_full_index.build(anthology)
    title_abs_full_metrics = evaluate_neural(title_abs_full_index, queries, k=k)

    print("[Neural] Field experiment results:")
    compare_table(
        {
            "title only": title_only_metrics,
            "title+abstract (baseline)": baseline_metrics,
            "title+abstract+full_text_truncated": title_abs_full_metrics,
        }
    )

    _print_subword_demo(baseline_index)
    return baseline_index


def run_long_docs(
    anthology,
    queries,
    baseline_index=None,
    model_name=MODEL_NAME,
    device=None,
    k=EVAL_K,
):
    """Part 2 step 2: IR overlapping word windows chunking + IR structure-aware chunking w/sections and paragraphs"""
    print("[Part 2][2/6] long documents: two IR methods")

    if baseline_index is None:
        baseline_index = NeuralIndex(
            model_name=model_name,
            fields=("title", "abstract"),
            device=device,
        )
        baseline_index.build(anthology)

    baseline_metrics = evaluate_neural(baseline_index, queries, k=k)

    # Method 1: fixed overlapping windows + max pooling
    # ~6 min on T4
    chunked_index = ChunkedIndex(
        model_name=model_name,
        encoder=baseline_index.encoder,
        max_words=220,
        overlap=40,
    )
    chunked_index.build(anthology)
    chunked_metrics = evaluate_neural(chunked_index, queries, k=k)

    # Method 2: structure-aware chunking + top-n aggregation
    # ~6 min on T4
    structured_index = StructuredChunkedIndex(
        model_name=model_name,
        encoder=baseline_index.encoder,
        max_words=220,
        min_words=80,
        target_words=180,
        top_n=3,
        aggregation="weighted",
    )
    structured_index.build(anthology)
    structured_metrics = evaluate_neural(structured_index, queries, k=k)

    print("[LongDocs][IR] whole-doc vs two long-doc methods:")
    compare_table(
        {
            "whole-doc neural (title+abstract)": baseline_metrics,
            "method-1: fixed windows + max": chunked_metrics,
            "method-2: structured + top3 weighted": structured_metrics,
        }
    )

    map_delta_1 = chunked_metrics["MAP@10"] - baseline_metrics["MAP@10"]
    f1_delta_1 = chunked_metrics["macro_F1"] - baseline_metrics["macro_F1"]
    map_delta_2 = structured_metrics["MAP@10"] - baseline_metrics["MAP@10"]
    f1_delta_2 = structured_metrics["macro_F1"] - baseline_metrics["macro_F1"]
    print(
        "[LongDocs][IR] Delta method-1 vs whole-doc: "
        f"MAP@10={map_delta_1:+.4f}, "
        f"macro_F1={f1_delta_1:+.4f}"
    )
    print(
        "[LongDocs][IR] Delta method-2 vs whole-doc: "
        f"MAP@10={map_delta_2:+.4f}, "
        f"macro_F1={f1_delta_2:+.4f}"
    )

    return structured_index


def run_query_rewriting(index=None, queries=None, model=None, tokenizer=None, k=EVAL_K):
    """Part 2 step 3: query rewriting (single rewrite + RRF-3)."""
    print("[Part 2][3/6] query rewriting: original vs 1-rewrite vs RRF-3")

    if model is None or tokenizer is None:
        print("[Rewrite] skipped: model/tokenizer not provided.")
        return None

    if queries is None:
        queries = answerable_queries(load_queries())

    if index is None:
        anthology = load_anthology()
        index = NeuralIndex(model_name=MODEL_NAME, fields=("title", "abstract"))
        index.build(anthology)

    original_ranked_lists = []
    single_rewrite_ranked_lists = []
    rrf3_ranked_lists = []

    for i, entry in enumerate(queries):
        query = entry["q"]

        original_ranked = index.retrieve(query, k=k)
        original_ids = index.acl_ids([idx for _, idx in original_ranked])
        original_ranked_lists.append(original_ids)

        rewrites = rewrite_query(query, n=3, model=model, tokenizer=tokenizer)
        rewrite_ranked_lists = []
        for rewritten_query in rewrites:
            rewritten_ranked = index.retrieve(rewritten_query, k=k)
            rewrite_ranked_lists.append(index.acl_ids([idx for _, idx in rewritten_ranked]))

        single_rewrite_ranked_lists.append(rewrite_ranked_lists[0] if rewrite_ranked_lists else original_ids)

        rrf3_inputs = [original_ids] + rewrite_ranked_lists[:3]
        if len(rrf3_inputs) > 1:
            rrf3_ranked_lists.append(rrf_fuse(rrf3_inputs, k=k))
        else:
            rrf3_ranked_lists.append(original_ids)

        if i < 2:
            print(f"[Rewrite][sample {i+1}] query: {query}")
            print(f"[Rewrite][sample {i+1}] rewrites: {rewrites}")

    results = {
        "original": evaluate_ranked_lists(original_ranked_lists, queries, k=k),
        "1 rewrite": evaluate_ranked_lists(single_rewrite_ranked_lists, queries, k=k),
        "RRF-3": evaluate_ranked_lists(rrf3_ranked_lists, queries, k=k),
    }

    print("[Rewrite] Retrieval results:")
    compare_table(results)
    return results


def run_hyde(*args, **kwargs):
    index = kwargs.get("index")
    queries = kwargs.get("queries")
    model = kwargs.get("model")
    tokenizer = kwargs.get("tokenizer")
    k = kwargs.get("k", EVAL_K)

    if index is None and len(args) > 0:
        index = args[0]
    if queries is None and len(args) > 1:
        queries = args[1]
    if model is None and len(args) > 2:
        model = args[2]
    if tokenizer is None and len(args) > 3:
        tokenizer = args[3]

    print("[Part 2][4/6] HyDE: hypothetical abstract retrieval")

    if model is None or tokenizer is None:
        print("[HyDE] skipped: model/tokenizer not provided.")
        return None

    if queries is None:
        queries = answerable_queries(load_queries())

    if index is None:
        anthology = load_anthology()
        index = NeuralIndex(model_name=MODEL_NAME, fields=("title", "abstract"))
        index.build(anthology)

    baseline_ranked_lists = []
    hyde_ranked_lists = []
    sample_hypothesis = None
    sample_query = None

    for i, entry in enumerate(queries):
        query = entry["q"]

        base_ranked = index.retrieve(query, k=k)
        baseline_ranked_lists.append(index.acl_ids([idx for _, idx in base_ranked]))

        if i == 0:
            hyde_ids, hypothesis = hyde_retrieve(
                query,
                index,
                model=model,
                tokenizer=tokenizer,
                k=k,
                return_hypothesis=True,
            )
            sample_query = query
            sample_hypothesis = hypothesis
        else:
            hyde_ids = hyde_retrieve(
                query,
                index,
                model=model,
                tokenizer=tokenizer,
                k=k,
            )

        hyde_ranked_lists.append(hyde_ids)

    results = {
        "neural baseline": evaluate_ranked_lists(baseline_ranked_lists, queries, k=k),
        "HyDE": evaluate_ranked_lists(hyde_ranked_lists, queries, k=k),
    }

    print("[HyDE] Retrieval results:")
    compare_table(results)

    if sample_query is not None and sample_hypothesis is not None:
        print("[HyDE] Sample hypothetical abstract")
        print(f"Query: {sample_query}")
        print(sample_hypothesis)

    return results


def run_prompt_engineering(*args, **kwargs):
    print("[Part 2] Prompt-engineering step not implemented yet.")


def run_metadata_probe(*args, **kwargs):
    print("[Part 2] Metadata probe step not implemented yet.")


def run_data_step():
    """Debug convenience: build cache and print quick dataset/query stats."""
    build_sample_if_missing()
    anthology = load_anthology()
    queries = load_queries()
    ans = answerable_queries(queries)

    print(f"[Part 2][data] anthology docs: {len(anthology)}")
    if len(anthology) > 0:
        print(f"[Part 2][data] first acl_id: {anthology[0].get('acl_id')}")
    print(f"[Part 2][data] queries: {len(queries)} | answerable: {len(ans)}")


def run_long_step():
    """Debug convenience: run long-doc step without loading the LLM."""
    build_sample_if_missing()  # ~10 s if cached
    anthology = load_anthology()
    queries = answerable_queries(load_queries())

    baseline_index = NeuralIndex(
        model_name=MODEL_NAME,
        fields=("title", "abstract"),
    )
    baseline_index.build(anthology)
    return run_long_docs(
        anthology,
        queries,
        baseline_index=baseline_index
    )


def run_neural_step():
    build_sample_if_missing()  # ~10 s if cached
    anthology = load_anthology()
    queries = answerable_queries(load_queries())
    return run_neural_baseline(anthology, queries)


def run_hyde_step():
    """Debug convenience: run HyDE end-to-end with lazy LM load."""
    build_sample_if_missing()  # ~10 s if cached
    anthology = load_anthology()
    queries = answerable_queries(load_queries())

    index = NeuralIndex(model_name=MODEL_NAME, fields=("title", "abstract"))
    index.build(anthology)

    model, tokenizer = _load_part2_lm()
    return run_hyde(index=index, queries=queries, model=model, tokenizer=tokenizer)


def run_part2_all(model=None, tokenizer=None):
    build_sample_if_missing()  # ~10 s if cached
    anthology = load_anthology()
    queries = answerable_queries(load_queries())

    _banner("Neural baseline + field experiment", "4 min on T4")
    index = run_neural_baseline(anthology, queries)  # returns NeuralIndex

    _banner("Long documents (chunked index + LM stuffing)", "20 min on T4")
    run_long_docs(anthology, queries, index)

    _banner("Query rewriting (single + RRF-3)", "20 min on T4")
    run_query_rewriting(index, queries, model, tokenizer)

    _banner("HyDE (98 hypothetical abstracts)", "15 min on T4")
    run_hyde(index, queries, model, tokenizer)

    _banner("Prompt engineering (cite / abstain / hardened)", "5 min on T4")
    run_prompt_engineering(index, queries, model, tokenizer)

    _banner("Metadata probe (no LM)", "30 s")
    run_metadata_probe(index)


STEPS = {
    "data": lambda m=None, t=None: run_data_step(),
    "neural": lambda m=None, t=None: run_neural_step(),
    "long": lambda m=None, t=None: run_long_step(),
    "rewrite": run_query_rewriting,
    "hyde": lambda m=None, t=None: run_hyde_step(),
    "prompt": run_prompt_engineering,
    "metadata": run_metadata_probe,
    "all": run_part2_all,
}


if __name__ == "__main__":  # debug convenience — NOT the reproduction path
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=list(STEPS), default="all")
    args = ap.parse_args()
    STEPS[args.step]()
