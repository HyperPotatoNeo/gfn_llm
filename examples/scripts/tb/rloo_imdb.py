import torch
from tqdm import tqdm
import numpy as np

from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
    pipeline
)
from trl import ModelConfig
from datasets import load_dataset

from trl.trainer.rloo_imdb_trainer import RLOOConfig, RLOOTrainer
from trl.core import LengthSampler

import wandb



class LengthSampler:
    def __init__(self, min_value, max_value):
        self.values = list(range(min_value, max_value))
    def __call__(self):
        return np.random.choice(self.values)

input_size = 8#LengthSampler(2, 8)
output_size = 16#LengthSampler(4, 16)


if __name__ == '__main__':
    parser = HfArgumentParser((RLOOConfig, ModelConfig))
    config, model_config = parser.parse_args_into_dataclasses()
    wandb.init(project='advantage-diffusion', entity='swish')
    #wandb.init()

    #dataset = build_dataset(model_config)

    policy = AutoModelForCausalLM.from_pretrained(
        "lvwerra/gpt2-imdb", trust_remote_code=model_config.trust_remote_code
    )
    # This is the reference model (frozen) for the KL divergence
    ref_policy = AutoModelForCausalLM.from_pretrained(
        "lvwerra/gpt2-imdb", trust_remote_code=model_config.trust_remote_code
    )
    reward_model = AutoModelForSequenceClassification.from_pretrained(
        "lvwerra/gpt2-imdb", trust_remote_code=model_config.trust_remote_code, num_labels=2
    )
    tokenizer = AutoTokenizer.from_pretrained("lvwerra/gpt2-imdb")
    tokenizer.pad_token = tokenizer.eos_token
    def tokenize(sample):
        sample["input_ids"] = tokenizer.encode(sample["review"])[:input_size]
        sample["query"] = tokenizer.decode(sample["input_ids"])
        return sample
    dataset = load_dataset("imdb", split="train")
    dataset = dataset.select(range(len(dataset)))
    dataset = dataset.rename_columns({"text": "review"})
    dataset = dataset.filter(lambda x: len(x["review"]) > 200, batched=False)
    dataset = dataset.map(tokenize, batched=False)
    dataset.set_format(type="torch")

    trainer = RLOOTrainer(
        config=config,
        tokenizer=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        reward_model=reward_model,
        train_dataset=dataset,
        eval_dataset=dataset
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    if config.push_to_hub:
        trainer.push_to_hub()
    trainer.generate_completions()