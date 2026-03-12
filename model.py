prompt = f"""

YOUR PROMPT GOES HERE

"""

irrelevant_context = """
Richard Gary Brautigan (January 30, 1935 – c. September 16, 1984)
was an American novelist, poet, and short story writer. A prolific writer,
he wrote throughout his life and published ten novels, two collections of
short stories, and four books of poetry. Brautigan's work has been published
both in the United States and internationally throughout Europe, Japan,
and China. He is best known for his novels Trout Fishing in America (1967),
In Watermelon Sugar (1968), and The Abortion: An Historical Romance 1966 (1971).
"""


model_id = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_id)

bnb_config = transformers.BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    quantization_config=bnb_config,
    device_map='auto'
)

input_string = prompt + retrieval_queries[0]

encoded_prompt = tokenizer(input_string, return_tensors="pt", add_special_tokens=False)
encoded_prompt = encoded_prompt.to("cuda")

generated_ids = model.generate(**encoded_prompt, max_new_tokens=1000, do_sample=True)
decoded = tokenizer.batch_decode(generated_ids)
print(decoded[0])