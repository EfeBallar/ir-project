import json
import torch
import datasets
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM
from tf_idf import (
    doc_to_text, tokenize_docs, train_bigram_model,
    vectorize, vectorize_query, build_vocab, dot_sparse
)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_PATH      = "./irse_documents_2026_recipes.parquet"
QUERIES_PATH   = "./irse_queries_2026_recipes.json"
MODEL_ID       = "mistralai/Mistral-7B-Instruct-v0.2"
K_GENERATION   = 5     # docs passed to LLM
K_EVALUATION   = 10    # docs used for IR metrics

# ── Load model and tokenizer ──────────────────────────────────────────────────
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map='auto'
)

print("Model loaded.\n")

# ── Load and index dataset ────────────────────────────────────────────────────
dataset = datasets.load_dataset("parquet", data_files=DATA_PATH)["train"]
gold_queries = json.load(open(QUERIES_PATH))

doc_texts      = [doc_to_text(doc) for doc in dataset]
tokenized_docs = tokenize_docs(doc_texts)
bigram         = train_bigram_model(tokenized_docs)
tokenized_docs = [bigram[doc] for doc in tokenized_docs]
vocab, idf, df = build_vocab(tokenized_docs, min_df=5, max_df_ratio=0.5, max_vocab=50000)
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
def generate(prompt, max_new_tokens=300):
    encoded = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)
    encoded = encoded.to("cuda")
    generated_ids = model.generate(**encoded, max_new_tokens=max_new_tokens, do_sample=True)
    decoded = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    answer = decoded[0].split("[/INST]")[-1].strip()
    return answer

# ── Experiment 1: Compare prompt variants ────────────────────────────────────
# Assignment bullet 6: "Give two or more sets of instructions"
print("EXPERIMENT 1 — Prompt variant comparison")
print("═" * 70)

# Queries to use for qualitative LLM analysis
DEMO_QUERIES = [
    "What temperature should I pre-heat my oven to when making chicken quesadillas?",
    "What is the most important ingredient in borscht?",
    "Can I make borscht vegetarian?",
    "What is mirepoix?",
    "Is beef Wellington the only kind of meaty Wellington? If not, what other kinds do you know?",
]

for query in DEMO_QUERIES[:3]:   # run on first 3 queries to save time
    retrieved = retrieve(query, k=K_GENERATION)
    print(f"\nQuery: {query}")
    print(f"Retrieved: {[doc.get('name') for _, doc in retrieved]}\n")
    for variant_name, prompt_fn in PROMPT_VARIANTS.items():
        prompt = prompt_fn(query, retrieved)
        answer = generate(prompt)
        print(f"── Prompt {variant_name} ──")
        print(f"{answer}\n")
    print("-" * 70)

# ── Experiment 2: Does LM understand the RAG task? ───────────────────────────
# Assignment bullet 7: "Does the LM understand the RAG task?"
# Test: give it irrelevant recipes and see if it says it can't answer
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
answer = generate(prompt)
print(f"Query: {test_query}")
print(f"LM response:\n{answer}")

# ── Experiment 3: Reasoning queries (not just recipe retrieval) ───────────────
# Assignment bullet 7: "queries that require reasoning across recipes"
# All 5 must NOT be of the form "recommend me recipes"
print("EXPERIMENT 3 — Reasoning queries")
print("═" * 70)

reasoning_queries = [
    "What temperature should I pre-heat my oven to when making chicken quesadillas?",
    "What is the most important ingredient in borscht?",
    "Is beef Wellington the only kind of meaty Wellington? If not, what other kinds do you know?",
    "Where does shakshuka come from?",
    "What are the three essential ingredients in a baked Alaska?",
]

for query in reasoning_queries:
    retrieved = retrieve(query, k=K_GENERATION)
    prompt    = prompt_b(query, retrieved)
    answer    = generate(prompt)
    print(f"\nQuery: {query}")
    print(f"Retrieved: {[doc.get('name') for _, doc in retrieved]}")
    print(f"Answer: {answer}")
    print("-" * 70)

# ── Experiment 4: Does LM use supplied recipes or its own knowledge? ──────────
# Assignment bullet 7: "Verify LM is basing its answer on supplied recipes"
# Method: compare answer with correct recipes vs. answer with no context
print("EXPERIMENT 4 — Grounding check (RAG vs. no context)")
print("═" * 70)

grounding_query = "What is a turducken?"

# With correct retrieved context
retrieved = retrieve(grounding_query, k=K_GENERATION)
answer_with_context = generate(prompt_b(grounding_query, retrieved))

# Without any context (bare LM knowledge)
bare_prompt = f"<s>[INST] {grounding_query} [/INST]"
answer_without_context = generate(bare_prompt)

print(f"\nQuery: {grounding_query}")
print(f"Retrieved: {[doc.get('name') for _, doc in retrieved]}")
print(f"\n── With RAG context ──\n{answer_with_context}")
print(f"\n── Without context (bare LM) ──\n{answer_without_context}")

# ── Experiment 5: Score sensitivity ──────────────────────────────────────────
# Assignment bullet 7: "Do you notice significant changes when you include scores?"
print("EXPERIMENT 5 — Score sensitivity")
print("═" * 70)

score_query = "What temperature should I bake chicken Wellington at?"
retrieved   = retrieve(score_query, k=K_GENERATION)

answer_no_scores   = generate(prompt_b(score_query, retrieved))
answer_with_scores = generate(prompt_b_with_scores(score_query, retrieved))

print(f"\nQuery: {score_query}")
print(f"Scores: {[round(s, 3) for s, _ in retrieved]}")
print(f"\n── Without scores ──\n{answer_no_scores}")
print(f"\n── With scores ──\n{answer_with_scores}")

"""
EXPERIMENT 1 — Prompt variant comparison
══════════════════════════════════════════════════════════════════════

Query: What temperature should I pre-heat my oven to when making chicken quesadillas?
Retrieved: ['quesadilla combos', 'bbq quesadillas', 'popovers pre 1940', 'chicken jalapeno quesadillas nuwave oven flavorwave', 'jillyray cheese quesadillas salsa']

── Prompt A (minimal) ──
To make chicken quesadillas, you should pre-heat your oven to 425 degrees Fahrenheit, as advised in Recipe 2 for BBQ quesadillas.

── Prompt B (structured) ──
The recipes provided suggest pre-heating the oven to 425 degrees Fahrenheit for making chicken quesadillas, according to Recipe 2: BBQ quesadillas.

── Prompt B+scores ──
Based on the provided recipes, you should pre-heat your oven to 425 degrees Fahrenheit when making chicken quesadillas, according to Recipe 2.

----------------------------------------------------------------------

Query: What is the most important ingredient in borscht?
Retrieved: ['somma summer krut weed borscht', 'borscht ii', 'chicken vegetable borscht', 'simple borscht', 'red onion borscht']

── Prompt A (minimal) ──
The most important ingredient in borscht varies among different recipes. In Recipe 1, it is the farmer sausage or smoked ham hock that contributes greatly to the flavor of the soup. In Recipe 2, it is the beets that give the soup its signature color and earthy taste. In Recipe 3, it is the combination of chicken, vegetables, and beans that make up the main ingredients. In Recipe 4, it is the beetroot that provides the base for the soup. In Recipe 5, it is the red onions that take the starring role. Therefore, there is no single most important ingredient for borscht as each recipe may have different key components.

── Prompt B (structured) ──
Based on the provided recipes, the most important ingredient in borscht can vary as some recipes include sausage or meat, while others do not. However, one common ingredient among all recipes is beets, which plays a significant role in giving borscht its distinctive color and flavor. Therefore, beets can be considered an essential ingredient in making a traditional borscht.

── Prompt B+scores ──
Based on the provided recipes, beets are a consistent and significant ingredient in borscht. Therefore, I would identify beets as one of the most important ingredients in borscht. However, it's important to note that other ingredients like potatoes, onions, and herbs like dill and parsley also play essential roles in various borscht recipes.

----------------------------------------------------------------------

Query: Can I make borscht vegetarian?
Retrieved: ['somma summer krut weed borscht', 'borscht ii', 'chicken vegetable borscht', 'borscht beet soup', 'simple borscht']

── Prompt A (minimal) ──
Yes, you can make borscht vegetarian by omitting the meat and using vegetable broth instead of meat stock in recipes where meat is used. For example, you can follow Recipe 4: borscht beet soup but leave out the pork sausage and use low sodium vegetable broth instead. Or, you can use Recipe 5: simple borscht which is already a vegetarian recipe. Recipe 2: borscht II could also be made vegetarian by omitting the beef stew meat and using vegetable broth instead of chicken stock.

── Prompt B (structured) ──
Yes, borscht can be made vegetarian by omitting the meat ingredients such as sausage or beef stew meat in the provided recipes. For example, Recipes 3 and 4 can be made vegetarian by not adding the chicken or pork sausage respectively. Recipe 5 does not contain any meat to begin with, making it already vegetarian. Adjustments to taste with herbs, sour cream, or yogurt can be made as desired.

── Prompt B+scores ──
Yes, you can make borscht vegetarian by omitting the meat ingredients in any of the recipes provided. For example, Recipe 4 "borscht beet soup" can be made vegetarian by not adding the pork sausage. Alternatively, you can replace the meat with vegetarian substitutes like tofu or tempeh, if the recipes allow for it. However, keep in mind that some recipes, like Recipe 1, include meat as a crucial part of the traditional borscht preparation and may not taste the same without it.

----------------------------------------------------------------------

EXPERIMENT 2 — Does the LM understand the RAG task?
  (feeding irrelevant recipes — LM should say it cannot answer)
══════════════════════════════════════════════════════════════════════
Query: What temperature should I bake chicken Wellington at?
LM response:
I'm unable to provide an answer to your question about baking temperature for chicken Wellington based on the provided context. The context only mentions information about the author Richard Gary Brautigan and his literary works. For baking instructions, you may refer to a reliable cookbook or recipe website.

══════════════════════════════════════════════════════════════════════
EXPERIMENT 3 — Reasoning queries
══════════════════════════════════════════════════════════════════════

Query: What temperature should I pre-heat my oven to when making chicken quesadillas?
Retrieved: ['quesadilla combos', 'bbq quesadillas', 'popovers pre 1940', 'chicken jalapeno quesadillas nuwave oven flavorwave', 'jillyray cheese quesadillas salsa']
Answer: To answer the question, I examined the recipes provided, and two of them (Recipe 2 and Recipe 4) mention pre-heating an oven to specific temperatures for making quesadillas. According to these recipes, the oven should be pre-heated to 425 degrees Fahrenheit for making BBQ quesadillas in Recipe 2, and for the chicken jalapeno quesadillas in the Nu-wave oven in Recipe 4.

Therefore, according to the recipes available, you should preheat your oven to 425 degrees Fahrenheit when making chicken quesadillas.
----------------------------------------------------------------------

Query: What is the most important ingredient in borscht?
Retrieved: ['somma summer krut weed borscht', 'borscht ii', 'chicken vegetable borscht', 'simple borscht', 'red onion borscht']
Answer: Based on the provided recipes, the most important ingredient in borscht varies as some recipes include sausage or ham hocks, while others use beets as the main ingredient. However, one common theme among all the recipes is the use of stock or broth as the base for the soup. Therefore, while not an inherent ingredient in every recipe, a rich and flavorful stock or broth serves as the foundation for a successful borscht.
----------------------------------------------------------------------

Query: Is beef Wellington the only kind of meaty Wellington? If not, what other kinds do you know?
Retrieved: ['pork wellington', 'beef wellington two', 'individual pork wellingtons', 'beef wellington ramsay style', 'individual beef scallop wellingtons rachael ray']
Answer: Based on the recipes provided, beef Wellington is not the only kind of meaty Wellington. There are also pork and possibly sea scallop variations. The pork Wellington recipe is included in the provided recipes, and there is a mention of a sea scallop variation in Recipe 5.
----------------------------------------------------------------------

Query: Where does shakshuka come from?
Retrieved: ['shakshuka yemani basted eggs', 'shakshuka swiss chard', 'shannon shakshuka', 'shakshuka', 'ragu shuka ragu']
Answer: Shakshuka is a dish that originated from North Africa and the Middle East, specifically Israel and Tunisia. The name "shakshuka" comes from the Arabic word "shakshuka" which means "a shake or agitation." The recipes provided above show variations of the dish, including tomato-based sauces with eggs cracked on top and cooked in the pan, some with chard or other vegetables, and others with different spices and ingredients like paprika, cumin, and feta cheese. However, the essential element of shakshuka remains the same: a sauce made with tomatoes, aromatics, and eggs, cooked together in a pan and often served with bread.
----------------------------------------------------------------------

Query: What are the three essential ingredients in a baked Alaska?
Retrieved: ['heavenly baked alaska', 'simple baked alaska', 'baked alaska salmon fillet mustard sauce', 'chocolate baked alaska', 'quick baked alaska']
Answer: The three essential ingredients in a Baked Alaska are:
1. Ice cream
2. Egg whites
3. Sugar (both for meringue)

The recipes provide sufficient information to answer the question.
----------------------------------------------------------------------

EXPERIMENT 4 — Grounding check (RAG vs. no context)
══════════════════════════════════════════════════════════════════════

Query: What is a turducken?
Retrieved: ['turducken thanksgiving sides', 'turducken roulade', 'turducken chicken duck turkey', 'turducken', 'umami stuffing']

── With RAG context ──
A turducken is a dish consisting of a deboned turkey stuffed with a deboned duck, which in turn is stuffed with a deboned chicken. The name is a portmanteau of "turkey," "duck," and "chicken." The deboning process is time-consuming and may be done the day before to save time. The turducken is then seasoned, stuffed with prepared stuffing, and slow-cooked. The result is a multi-layered bird with each meat and flavor blending together. The recipes above provide detailed instructions for deboning and assembling a turducken, as well as for preparing additional side dishes, such as mashed potatoes and sweet potatoes, and Cajun-inspired stuffing and rice dressing.

── Without context (bare LM) ──
A turducken is a culinary dish that consists of a deboned turkey stuffed with a duck, which in turn is stuffed with a chicken. The chicken may also be stuffed with herbs, onions, and other ingredients for added flavor. The entire bird suite is then roasted, resulting in a multi-layered and richly flavored main course. It is a combination of three poultry dishes in one, and is considered an intricate and indulgent holiday meal.

EXPERIMENT 5 — Score sensitivity
══════════════════════════════════════════════════════════════════════

Query: What temperature should I bake chicken Wellington at?
Scores: [0.362, 0.341, 0.315, 0.302, 0.291]

── Without scores ──
The chicken Wellington recipe in Recipe 1 states to preheat the oven to 180c / 160c fan forced.

── With scores ──
The recipe with the highest relevance score, Recipe 1, specifies that the oven should be preheated to 180c / 160c fan forced for baking chicken Wellington.
"""

  if __name__ == "__main__":
      print("[Part 1/2] running Part 1 experiments (~6 min on T4)")
      run_part1_experiments()
      print("[Part 2/2] running Part 2 experiments (~60 min on T4)")
      from part2 import run_part2_all
      run_part2_all(model=model, tokenizer=tokenizer)