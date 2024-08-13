# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.

# For dataset details visit: https://huggingface.co/datasets/samsum

import copy
import datasets


'''
{
    'game_id': 'game_id',
    'moves': '...',
    'winner': 'white' or 'black' or 'draw',
}
'''

def get_chess_dataset(dataset_config, tokenizer, split="train"):
    dataset = datasets.load_dataset("conacts/stockfish_dataset", split=split)
    dataset = dataset.select(range(1000))

    def apply_prompt_template(sample):
        return {
                "moves" : sample["moves"],
        }

    dataset = dataset.map(apply_prompt_template)

    def tokenize_add_label(sample):
        moves = tokenizer.encode(tokenizer.bos_token + sample["moves"], add_special_tokens=False)

        sample = {
            "input_ids": moves,
            "attention_mask" : [1] * len(moves),
            "labels": moves,
        }

        return sample

    dataset = dataset.map(tokenize_add_label, remove_columns=list(dataset.features))

    return dataset
