import multiprocessing
import shutil
import wandb

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
import os
from trl.trainer import ModelConfig
from trl.trainer.tb_trainer_reasoning_vllm  import TBConfig, TBTrainerReasoning

from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE
import re
import torch
from vllm import LLM, SamplingParams
#from utils import WandbLogModelConfig

"""
module unload anaconda
echo "loading modules"
module load python/3.10 cudatoolkit/12.3.2

echo "loading env"
cd $HOME/johan_phd/
source llm_gfn_git/bin/activate

echo "running script.."
cd $HOME/scratch/gfn_llm/

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    --num_processes 2 \
    examples/scripts/tb/tb_GSM8K_vllm.py \
    --learning_rate 2e-6 \
    --output_dir models/GSM8K/tb_GSM8K_vllm \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 16 \
    --total_episodes 59760 \
    --model_name_or_path microsoft/rho-math-1b-v0.1 \
    --sft_model_path realtreetune/rho-1b-sft-GSM8K \
    --non_eos_penalty \
    --stop_token eos \
    --response_length 512 \
    --sanity_check

python3 examples/scripts/tb/tb_GSM8K_vllm.py \
    --learning_rate 3e-6 \
    --output_dir models/GSM8K/tb_GSM8K_vllm \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --total_episodes 30000 \
    --model_name_or_path microsoft/rho-math-1b-v0.1 \
    --sft_model_path realtreetune/rho-1b-sft-GSM8K \
    --non_eos_penalty \
    --stop_token eos \
    --response_length 512 \
    --sanity_check
"""


if __name__ == "__main__":
    # wandb.init(project='trl')
    parser = HfArgumentParser((TBConfig, ModelConfig))
    config, model_config = parser.parse_args_into_dataclasses()
    # remove output_dir if exists
    shutil.rmtree(config.output_dir, ignore_errors=True)

    ################
    # Model & Tokenizer
    ################c
    tokenizer = AutoTokenizer.from_pretrained(
        config.sft_model_path,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})

    if tokenizer.eos_token_id is None:
        tokenizer.add_special_tokens({"eos_token": "<eos>"})

    ref_policy = AutoModelForCausalLM.from_pretrained(
        config.sft_model_path, trust_remote_code=model_config.trust_remote_code
    )
    policy = AutoModelForCausalLM.from_pretrained(
        config.sft_model_path, trust_remote_code=model_config.trust_remote_code
    )
    # Align padding tokens between tokenizer and model
    policy.config.pad_token_id = tokenizer.pad_token_id
    policy.config.eos_token_id = tokenizer.eos_token_id
    
    # Align padding tokens between tokenizer and model
    ref_policy.config.pad_token_id = tokenizer.pad_token_id
    ref_policy.config.eos_token_id = tokenizer.eos_token_id

    ################
    # Dataset
    ################
    cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
    raw_datasets = load_dataset("openai/gsm8k", 'main', cache_dir=cache_dir)
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]
    
    def parse_number(value):
        value = value.strip()
        value = value.replace(',', '')
        return float(value)
    
    def data_processing(query):
        question_template = f'[MATH_TASK] Problem:\n{query}\n\nSolution:'
        return question_template
    
    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            data_pross = data_processing(element["question"])
            input_ids = tokenizer(
                data_pross,
                padding=False,
            )
            number = parse_number(element["answer"].split('####')[1])
            return {"input_ids": input_ids['input_ids'], 
                    "lengths": len(input_ids['input_ids']), 
                    "response_ids":number,
                    }

        return dataset.map(
            tokenize,
            remove_columns=dataset.column_names,
            num_proc=1 if config.sanity_check else multiprocessing.cpu_count(),
            load_from_cache_file=not config.sanity_check,
        )
    
    train_dataset = prepare_dataset(train_dataset, tokenizer) #7470 samples
    eval_dataset = prepare_dataset(eval_dataset, tokenizer) #1320

    ################
    # Training
    ################
    trainer = TBTrainerReasoning(
        config=config,
        tokenizer=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        #reward_model=reward_model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    print('===Saving Model ....')
    trainer.save_model(config.output_dir)
    print('===Pushing to hub ....')
    if config.push_to_hub:
        trainer.push_to_hub()
    print('===Generating completions ....')
    trainer.generate_completions()
