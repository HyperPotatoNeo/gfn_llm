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
from trl.trainer import ModelConfig #(Ask to SIDDARTH)
from trl.trainer.rloo_trainer_reasoning_vllm import RLOOConfig, RLOOTrainerReasoning

from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE
import re
import torch
from vllm import LLM, SamplingParams
from utils import WandbLogModelConfig

"""
module unload anaconda
echo "loading modules"
module load python/3.10 cudatoolkit/12.3.2

echo "loading env"
cd $HOME/johan_phd/
source llm_gfn_git/bin/activate

echo "running script.."
cd $HOME/scratch/gfn_llm/

python3 examples/scripts/rloo/rloo_GSM8K_vllm.py \
    --learning_rate 3e-6 \
    --output_dir models/GSM8K/ppo \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --total_episodes 30000 \
    --model_name_or_path microsoft/rho-math-1b-v0.1 \
    --sft_model_path realtreetune/rho-1b-sft-GSM8K \
    --non_eos_penalty \
    --stop_token eos \
    --response_length 1024 \
    --sanity_check
    
"""


if __name__ == "__main__":
    # wandb.init(project='trl')
    parser = HfArgumentParser((RLOOConfig, ModelConfig))
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
    
    #torch.cuda.empty_cache() 
    
    device_policy = torch.device("cuda:0")  # Assign to GPU 1
    ref_policy = AutoModelForCausalLM.from_pretrained(
        config.sft_model_path, trust_remote_code=model_config.trust_remote_code
    )#.to(device_policy)  # Move to GPU 1

    policy = AutoModelForCausalLM.from_pretrained(
        config.sft_model_path, trust_remote_code=model_config.trust_remote_code
    )#.to(device_policy)
    #ref_policy = policy  

    # Align padding tokens between tokenizer and model
    policy.config.pad_token_id = tokenizer.pad_token_id
    policy.config.eos_token_id = tokenizer.eos_token_id
    
    # Align padding tokens between tokenizer and model
    ref_policy.config.pad_token_id = tokenizer.pad_token_id
    ref_policy.config.eos_token_id = tokenizer.eos_token_id
    
    # Start serving the model with 50% GPU memory utilization
    # os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # # Clear GPU memory before starting
    # torch.cuda.empty_cache()
    # torch.cuda.synchronize()
    # llm = LLM(
    #     model=config.sft_model_path,       # Path to the model
    #     # enforce_eager=True,                # Disable CUDA graphs for reduced memory usage
    #     max_model_len=512,                # Example: set max input length for the model ->1024
    #     gpu_memory_utilization=0.9,        # Limit GPU memory usage (optional)
    #     tensor_parallel_size=1,            # Example: set tensor parallelism (optional)
    #     device="cuda:3",
    # )

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
        #question_template = f'{bos_token} [MATH_TASK] Problem: {query} Solution:'
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
    
    train_dataset = prepare_dataset(train_dataset, tokenizer)
    eval_dataset = prepare_dataset(eval_dataset, tokenizer)

    ################
    # Training
    ################
    # Reward model removed!
    trainer = RLOOTrainerReasoning(
        config=config,
        tokenizer=tokenizer,
        policy=policy,
        # vllm_policy=llm,
        ref_policy=ref_policy,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        callbacks=[WandbLogModelConfig(model_config)]
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    if config.push_to_hub:
        trainer.push_to_hub()
    trainer.generate_completions()
