import json, torch, datasets, transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from part_1.tf_idf import (
    doc_to_text, tokenize_docs, train_bigram_model,
    vectorize, vectorize_query, build_vocab, dot_sparse
)

# ── Config ────────────────────────────────────────────────────────────────────
PART_1_DATA_PATH = "part_1/irse_documents_2026_recipes.parquet"
PART_1_QUERIES_PATH = "part_1/irse_queries_2026_recipes.json"
MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.2"
K_GENERATION = 5  # docs passed to LLM
K_EVALUATION = 10  # docs used for IR metrics

# ── Load and index dataset ────────────────────────────────────────────────────
dataset = datasets.load_dataset("parquet", data_files=PART_1_DATA_PATH)["train"]
gold_queries = json.load(open(PART_1_QUERIES_PATH))

doc_texts      = [doc_to_text(doc) for doc in dataset]
tokenized_docs = tokenize_docs(doc_texts)
bigram         = train_bigram_model(tokenized_docs)
tokenized_docs = [bigram[doc] for doc in tokenized_docs]
vocab, idf, df = build_vocab(tokenized_docs, min_df=5, max_df_ratio=0.5, max_vocab=50000)
# ~30 s on T4
doc_vectors    = [vectorize(tokens, vocab, idf) for tokens in tokenized_docs]
print("Indexing complete.\n")

# ── Retrieval ─────────────────────────────────────────────────────────────────
def retrieve(query, k=K_GENERATION):
    """Return top-k (score, doc) pairs for a query."""
    query_vec = vectorize_query(query, bigram, vocab, idf)
    if not query_vec:
        return []
    scores = [(dot_sparse(query_vec, vec), idx) for idx, vec in enumerate(doc_vectors)]
    scores.sort(reverse=True)
    return [(score, dataset[idx]) for score, idx in scores[:k]]

# ── Recipe formatters ─────────────────────────────────────────────────────────
def format_recipe(doc, rank=None, score=None, include_score=False):
    """
    Format a recipe doc for inclusion in a prompt.
    Steps are included here (unlike IR indexing) because the LLM needs procedural detail to answer reasoning queries.
    """
    header = f"Recipe {rank}" if rank else "Recipe"
    if include_score and score is not None:
        header += f" (relevance score: {score:.3f})"
    steps = doc.get("steps", "") or ""
    if isinstance(steps, list):
        steps = " ".join(steps)
    return (
        f"{header}: {doc.get('name', '')}\n"
        f"Description: {doc.get('description', '') or 'N/A'}\n"
        f"Ingredients: {doc.get('ingredients', '') or 'N/A'}\n"
        f"Steps: {steps or 'N/A'}"
    )

def format_recipes(retrieved, include_score=False):
    parts = []
    for rank, (score, doc) in enumerate(retrieved, start=1):
        parts.append(format_recipe(doc, rank=rank, score=score, include_score=include_score))
    return "\n\n".join(parts)

# ── Prompt templates ──────────────────────────────────────────────────────────
# Mistral-7B-Instruct uses [INST]...[/INST] chat template.

def prompt_a(query, retrieved):
    """
    Prompt A — Minimal
    Tests whether bare instructions are sufficient.
    """
    recipes = format_recipes(retrieved, include_score=False)
    return (
        f"<s>[INST] Answer the following question using only the provided recipes.\n\n"
        f"Question: {query}\n\n"
        f"Recipes:\n{recipes}\n\n"
        f"Answer: [/INST]"
    )

def prompt_b(query, retrieved):
    """
    Prompt B — Structured, role-assigned
    Tests whether richer instructions improve grounding.
    """
    recipes = format_recipes(retrieved, include_score=False)
    return (
        f"<s>[INST] You are a culinary assistant. You will be given a question and a set "
        f"of relevant recipes retrieved from a cookbook database. Your task is to answer "
        f"the question by reasoning across the provided recipes. Do not use any knowledge "
        f"outside of the provided recipes. If the recipes do not contain enough information "
        f"to answer the question, say so explicitly.\n\n"
        f"Question: {query}\n\n"
        f"Retrieved recipes:\n{recipes}\n\n"
        f"Based only on the recipes above, answer the question concisely. [/INST]"
    )

def prompt_b_with_scores(query, retrieved):
    """
    Prompt B with relevance scores included.
    Tests whether the LM uses scores to weight its answer.
    """
    recipes = format_recipes(retrieved, include_score=True)
    return (
        f"<s>[INST] You are a culinary assistant. You will be given a question and a set "
        f"of relevant recipes retrieved from a cookbook database, each annotated with a "
        f"relevance score between 0 and 1 (higher = more relevant). Your task is to answer "
        f"the question by reasoning across the provided recipes, giving more weight to "
        f"higher-scored recipes. Do not use any knowledge outside of the provided recipes. "
        f"If the recipes do not contain enough information to answer the question, say so "
        f"explicitly.\n\n"
        f"Question: {query}\n\n"
        f"Retrieved recipes:\n{recipes}\n\n"
        f"Based only on the recipes above, answer the question concisely. [/INST]"
    )

PROMPT_VARIANTS = {
    "A (minimal)"         : prompt_a,
    "B (structured)"      : prompt_b,
    "B+scores"            : prompt_b_with_scores,
}

# ── LLM inference ─────────────────────────────────────────────────────────────
def generate(prompt, model, tokenizer, max_new_tokens=300):
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = encoded.to("cuda")
    generated_ids = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    answer = decoded[0].split("[/INST]")[-1].strip()
    return answer

# Queries to use for qualitative LLM analysis (Part 1)
PART_1_DEMO_QUERIES = [
    "What temperature should I pre-heat my oven to when making chicken quesadillas?",
    "What is the most important ingredient in borscht?",
    "Can I make borscht vegetarian?",
    "What is mirepoix?",
    "Is beef Wellington the only kind of meaty Wellington? If not, what other kinds do you know?",
]

PART_1_REASONING_QUERIES = [
    "What temperature should I pre-heat my oven to when making chicken quesadillas?",
    "What is the most important ingredient in borscht?",
    "Is beef Wellington the only kind of meaty Wellington? If not, what other kinds do you know?",
    "Where does shakshuka come from?",
    "What are the three essential ingredients in a baked Alaska?",
]


def run_part1_experiments(model, tokenizer):
    # ── Experiment 1: Compare prompt variants ────────────────────────────────
    # Assignment bullet 6: "Give two or more sets of instructions"
    print("[Part 1][Step 1/5] Prompt variant comparison (~3 min on T4)")
    print("EXPERIMENT 1 — Prompt variant comparison")
    print("═" * 70)

    # ~3 min on T4
    for query in PART_1_DEMO_QUERIES[:3]:  # run on first 3 queries to save time
        retrieved = retrieve(query, k=K_GENERATION)
        print(f"\nQuery: {query}")
        print(f"Retrieved: {[doc.get('name') for _, doc in retrieved]}\n")
        for variant_name, prompt_fn in PROMPT_VARIANTS.items():
            prompt = prompt_fn(query, retrieved)
            answer = generate(prompt, model, tokenizer)
            print(f"── Prompt {variant_name} ──")
            print(f"{answer}\n")
        print("-" * 70)

    # ── Experiment 2: Does LM understand the RAG task? ───────────────────────
    # Assignment bullet 7: "Does the LM understand the RAG task?"
    # Test: give it irrelevant recipes and see if it says it can't answer
    print("[Part 1][Step 2/5] RAG understanding check (~30 s on T4)")
    print("EXPERIMENT 2 — Does the LM understand the RAG task?")
    print("  (feeding irrelevant recipes — LM should say it cannot answer)")
    print("═" * 70)

    # Experiment 2: feed completely irrelevant non-recipe context
    test_query = "What temperature should I bake chicken Wellington at?"
    irrelevant_context = """
Richard Gary Brautigan (January 30, 1935 – c. September 16, 1984)
was an American novelist, poet, and short story writer. A prolific writer,
he wrote throughout his life and published ten novels, two collections of
short stories, and four books of poetry...
"""
    # use their provided irrelevant context instead of wrong recipes
    prompt = f"<s>[INST] Answer the following question using only the provided context.\n\nQuestion: {test_query}\n\nContext:\n{irrelevant_context}\n\nAnswer: [/INST]"
    answer = generate(prompt, model, tokenizer)
    print(f"Query: {test_query}")
    print(f"LM response:\n{answer}")

    # ── Experiment 3: Reasoning queries (not just recipe retrieval) ──────────
    # Assignment bullet 7: "queries that require reasoning across recipes"
    # All 5 must NOT be of the form "recommend me recipes"
    print("[Part 1][Step 3/5] Reasoning queries (~2 min on T4)")
    print("EXPERIMENT 3 — Reasoning queries")
    print("═" * 70)

    # ~2 min on T4
    for query in PART_1_REASONING_QUERIES:
        retrieved = retrieve(query, k=K_GENERATION)
        prompt = prompt_b(query, retrieved)
        answer = generate(prompt,model, tokenizer)
        print(f"\nQuery: {query}")
        print(f"Retrieved: {[doc.get('name') for _, doc in retrieved]}")
        print(f"Answer: {answer}")
        print("-" * 70)

    # ── Experiment 4: Does LM use supplied recipes or its own knowledge? ─────
    # Assignment bullet 7: "Verify LM is basing its answer on supplied recipes"
    # Method: compare answer with correct recipes vs. answer with no context
    print("[Part 1][Step 4/5] Grounding check (~30 s on T4)")
    print("EXPERIMENT 4 — Grounding check (RAG vs. no context)")
    print("═" * 70)

    grounding_query = "What is a turducken?"

    # ~30 s on T4
    # With correct retrieved context
    retrieved = retrieve(grounding_query, k=K_GENERATION)
    answer_with_context = generate(prompt_b(grounding_query, retrieved),model, tokenizer)

    # Without any context (bare LM knowledge)
    bare_prompt = f"<s>[INST] {grounding_query} [/INST]"
    answer_without_context = generate(bare_prompt,model, tokenizer)

    print(f"\nQuery: {grounding_query}")
    print(f"Retrieved: {[doc.get('name') for _, doc in retrieved]}")
    print(f"\n── With RAG context ──\n{answer_with_context}")
    print(f"\n── Without context (bare LM) ──\n{answer_without_context}")

    # ── Experiment 5: Score sensitivity ──────────────────────────────────────
    # Assignment bullet 7: "Do you notice significant changes when you include scores?"
    print("[Part 1][Step 5/5] Score sensitivity (~30 s on T4)")
    print("EXPERIMENT 5 — Score sensitivity")
    print("═" * 70)

    score_query = "What temperature should I bake chicken Wellington at?"

    # ~30 s on T4
    retrieved = retrieve(score_query, k=K_GENERATION)
    answer_no_scores = generate(prompt_b(score_query, retrieved),model, tokenizer)
    answer_with_scores = generate(prompt_b_with_scores(score_query, retrieved),model, tokenizer)

    print(f"\nQuery: {score_query}")
    print(f"Scores: {[round(s, 3) for s, _ in retrieved]}")
    print(f"\n── Without scores ──\n{answer_no_scores}")
    print(f"\n── With scores ──\n{answer_with_scores}")


if __name__ == "__main__":
    # Load model and tokenizer
    print("Loading LM and tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    bnb_config = transformers.BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
    )
    print("LM loaded.\n")
    
    print("[Part 1/2] running Part 1 experiments (~6 min on T4)")
    run_part1_experiments(model,tokenizer)
    print("[Part 2/2] running Part 2 experiments (~60 min on T4)")
    from part_2.part2 import run_part2_experiments

    run_part2_experiments(model=model, tokenizer=tokenizer)
