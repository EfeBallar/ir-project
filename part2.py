import argparse
from part2_data import build_sample_if_missing, load_anthology, load_queries, answerable_queries
from neural_retrieval import NeuralIndex
from long_docs import ChunkedIndex, top_chunks_for_prompt
from part2_metrics import evaluate_neural, evaluate_ranked_lists, compare_table

def _banner(label, eta):
    print(f"\n[Part 2] {label}  (~{eta})")

def run_part2_all(model, tokenizer):
    build_sample_if_missing()                         # ~10 s if cached
    anthology = load_anthology()
    queries   = answerable_queries(load_queries())

    _banner("Neural baseline + field experiment", "4 min on T4")
    index = run_neural_baseline(anthology, queries)   # returns NeuralIndex

    _banner("Long documents (chunked index + LM stuffing)", "10 min on T4")
    chunked = run_long_docs(anthology, queries, index, model, tokenizer)

    _banner("Query rewriting (single + RRF-3, RRF-5)", "20 min on T4")
    run_query_rewriting(index, queries, model, tokenizer)

    _banner("HyDE (98 hypothetical abstracts)", "15 min on T4")
    run_hyde(index, queries, model, tokenizer)

    _banner("Prompt engineering (cite / abstain / hardened)", "5 min on T4")
    run_prompt_engineering(index, queries, model, tokenizer)

    _banner("Metadata probe (no LM)", "30 s")
    run_metadata_probe(index)

STEPS = {
    "data":     lambda m=None, t=None: build_sample_if_missing(),
    "neural":   run_neural_baseline,
    "long":     run_long_docs,
    "rewrite":  run_query_rewriting,
    "hyde":     run_hyde,
    "prompt":   run_prompt_engineering,
    "metadata": run_metadata_probe,
    "all":      run_part2_all,
}

if __name__ == "__main__":   # debug convenience — NOT the reproduction path
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", choices=list(STEPS), default="all")
    args = ap.parse_args()
    # Lazy-load the LM only for steps that need it.
    STEPS[args.step]()