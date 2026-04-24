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

"""
Unlike last time, we will not provide starter code for working with document embedding models. 
We trust that you are familiar enough with the transformers API to do so for this part of the project, 
after having completed Part I. If you are unsure of where to start, we recommend taking a look at the 
Massive Text Embedding Benchmark (MTEB) leaderboard,which ranks text embedding models based on performance. 
Note that not all models listed on leaderboard are open source and thus require an API access key 
(e.g. the OpenAI or Cohere embedding models). Also, if you find the huggingface software stack to be cumbersome, 
feel free to try the sentence-transformers library.
"""

# TODO: 
# 
# 