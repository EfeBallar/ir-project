# source venv/bin/activate
import json, datasets
from tf_idf import retrieve_top_k_docs, doc_to_text, tokenize_docs, train_bigram_model, vectorize, vectorize_query, build_vocab, dot_sparse

DATA_PATH = "./irse_documents_2026_recipes.parquet"

dataset = datasets.load_dataset("parquet", data_files=DATA_PATH)["train"]
gold_queries = json.load(open("./irse_queries_2026_recipes.json", "r"))

retrieval_queries = [
    "cajun style gumbo with an easy roux",
    "shrimp tacos recipe for tonight",
    "vegetarian lasagna but easy",
    "spageti and meatballs?",
    "lunch baked at 200 °C",
]

doc_texts = [doc_to_text(doc) for doc in dataset]
tokenized_docs = tokenize_docs(doc_texts)
bigram = train_bigram_model(tokenized_docs)
tokenized_docs = [bigram[doc] for doc in tokenized_docs]
# "olive oil" → "olive_oil" as a single token

vocab, idf, df = build_vocab(tokenized_docs, min_df=5, max_df_ratio=0.5, max_vocab=50000)
print("Built vocab")
doc_vectors = [vectorize(tokens, vocab, idf) for tokens in tokenized_docs]
print("Finished vectorizing docs")

# nearest neighbors search
results = retrieve_top_k_docs(retrieval_queries, doc_vectors, vocab, idf, bigram,dataset, k=3,threshold=0.3)
# 0.30 was chosen by eyeballing 5 queries, which is not rigorous! Validate!
