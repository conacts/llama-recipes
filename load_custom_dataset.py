from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_recipes.configs.datasets import chess_dataset 
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset

tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B")


train_dataset = get_preprocessed_dataset(tokenizer, chess_dataset, 'train')

print(train_dataset[0])

