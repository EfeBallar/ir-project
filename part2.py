import json
import datasets

R_NUMBER_SEED = 1072381 # student number
DOCS_TO_ADD = 2000

query_documents = datasets.load_dataset("parquet", data_files="./acl_anthology_queries.parquet")["train"]
all_documents = datasets.load_dataset("parquet", data_files="./acl_anthology_full.parquet")["train"]
# Shuffle with seed and take only n docs
random_documents = all_documents.shuffle(seed=R_NUMBER_SEED).take(DOCS_TO_ADD)
# Concatenate relevant documents with random sample and shuffle again
anthology_sample = datasets.concatenate_datasets([query_documents, random_documents]).shuffle(seed=R_NUMBER_SEED)
# Export to Parquet to avoid downloading full anthology
anthology_sample.to_parquet("./anthology_sample.parquet")

print(anthology_sample[0])
#{'acl_id': 'W02-1406', 'abstract': ".....", 'title': 'Probabilistic Named Entity Verification'}


# Like last time, we provide a queries `json` object, which you can utilize 
# to tune your retriever. The retrieved documents ids correspond to the `acl_id`
# field in the anthology dataset.

with open("./acl_anthology_queries.json", "r") as handle:
    queries = json.load(handle)
    for k, v in queries["queries"][0].items():
        print(f"{k.upper()}: {v}\n")
        
"""

Q: Which versions of the Morfessor tokenizer have been proposed in the literature?

R: ['W02-0603', 'E14-2006', 'C14-1111', '2020.lrec-1.486']

A: Baseline (MDL and ML), CatMap, CatML, FlatCat, EM+Prune

"""

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