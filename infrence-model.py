from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel


tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3.1-8B")
adapter_model = PeftModel.from_pretrained(model, './models/meta-llama-chess-2024-08-10')

input_text = "e4 d5 exd5 Nf6 d4 Qxd5 Nc3 Qd6 Nf3 Bd7"
inputs = tokenizer(input_text, return_tensors="pt")

output_sequences = adapter_model.generate(
    input_ids=inputs['input_ids'],
    max_length=50,  # Adjust the max length as needed
    num_return_sequences=1,  # Number of sequences to return
    temperature=0.7,  # Adjust temperature for more/less randomness
    top_k=50,  # Sampling from top_k tokens
    top_p=0.9,  # Nucleus sampling
    do_sample=True,  # Set to True to sample from the distribution
)

generated_text = tokenizer.decode(output_sequences[0], skip_special_tokens=True)

print("Generated Text: ", generated_text)
