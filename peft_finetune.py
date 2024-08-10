import torch
from transformers import LlamaForCausalLM, AutoTokenizer, AutoConfig
from llama_recipes.configs import train_config as TRAIN_CONFIG
import os
from datetime import datetime

output_dir = "./meta-llama-chess-" + datetime.now().strftime("%Y-%m-%d")

train_config = TRAIN_CONFIG()
if os.path.exists(output_dir):
    train_config.model_name = output_dir
    print("Using dir: ", output_dir)
else:
    train_config.model_name = "meta-llama/Meta-Llama-3.1-8B"
    print("Pulling from HF")

train_config.num_epochs = 100
train_config.run_validation = False
train_config.gradient_accumulation_steps = 4
train_config.batch_size_training = 1
train_config.lr = 3e-4
train_config.use_fast_kernels = True
train_config.use_fp16 = True
train_config.context_length = 1024 if torch.cuda.get_device_properties(0).total_memory < 16e9 else 2048 # T4 16GB or A10 24GB
train_config.batching_strategy = "packing"
train_config.output_dir = output_dir

import wandb
wandb_run = wandb.init(project="chess-gpt", name="8-9-2024", config=train_config)

from transformers import BitsAndBytesConfig
config = BitsAndBytesConfig(
    load_in_8bit=True,
)


model = LlamaForCausalLM.from_pretrained(
            train_config.model_name,
            device_map="auto",
            quantization_config=config,
            use_cache=False,
            attn_implementation="sdpa" if train_config.use_fast_kernels else None,
            torch_dtype=torch.float16,
        )


config = AutoConfig.from_pretrained(train_config.model_name)
tokenizer = AutoTokenizer.from_pretrained(train_config.model_name)
tokenizer.pad_token = tokenizer.eos_token

from llama_recipes.configs.datasets import chess_dataset
from llama_recipes.data.concatenator import ConcatDataset
from llama_recipes.utils.config_utils import get_dataloader_kwargs
from llama_recipes.utils.dataset_utils import get_preprocessed_dataset

train_dataset = get_preprocessed_dataset(tokenizer, chess_dataset, 'train')

train_dl_kwargs = get_dataloader_kwargs(train_config, train_dataset, tokenizer, "train")

if train_config.batching_strategy == "packing":
        train_dataset = ConcatDataset(train_dataset, chunk_size=train_config.context_length)

# Create DataLoaders for the training and validation dataset
train_dataloader = torch.utils.data.DataLoader(
    train_dataset,
    num_workers=train_config.num_workers_dataloader,
    pin_memory=True,
    **train_dl_kwargs,
)

from peft import get_peft_model, prepare_model_for_kbit_training, LoraConfig
from dataclasses import asdict
from llama_recipes.configs import lora_config as LORA_CONFIG

lora_config = LORA_CONFIG()
lora_config.r = 8
lora_config.lora_alpha = 32
lora_dropout: float=0.01

peft_config = LoraConfig(**asdict(lora_config))

model = prepare_model_for_kbit_training(model)
model = get_peft_model(model, peft_config)

import torch.optim as optim
from llama_recipes.utils.train_utils import train
from torch.optim.lr_scheduler import StepLR

model.train()

optimizer = optim.AdamW(
            model.parameters(),
            lr=train_config.lr,
            weight_decay=train_config.weight_decay,
        )
scheduler = StepLR(optimizer, step_size=1, gamma=train_config.gamma)

try:
    results = train(
        model,
        train_dataloader,
        None,
        tokenizer,
        optimizer,
        scheduler,
        train_config.gradient_accumulation_steps,
        train_config,
        None,
        None,
        None,
        wandb_run=wandb_run,
    )
except Exception as e:
    print(f"Training interrupted with error: {e}")
    # Save model weights before exiting
    print("Saving Models due to exception: ", train_config.output_dir)
    model.save_pretrained(train_config.output_dir)
    tokenizer.save_pretrained(train_config.output_dir)
    config.save_pretrained(train_config.output_dir)
    raise e
finally:
    print("Saving final model after training completion.")
    model.save_pretrained(train_config.output_dir)
    tokenizer.save_pretrained(train_config.output_dir)
    config.save_pretrained(train_config.output_dir)

