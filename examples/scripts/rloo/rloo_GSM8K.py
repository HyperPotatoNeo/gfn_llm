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
from trl.trainer.rloo_trainer_reasoning import RLOOConfig, RLOOTrainerReasoning

from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE
import re

"""
module unload anaconda
echo "loading modules"
module load python/3.10 cudatoolkit/12.3.2

echo "loading env"
cd $HOME/johan_phd/
source llm_gfn_git/bin/activate

echo "running script.."
cd $HOME/scratch/gfn_llm/

python3 examples/scripts/rloo/rloo_GSM8K.py \
    --learning_rate 3e-6 \
    --output_dir models/GSM8K/ppo \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --total_episodes 30000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped  \
    --sft_model_path EleutherAI/pythia-1b-deduped \
    --non_eos_penalty \
    --stop_token eos \
    --response_length 53 \
    --sanity_check

python3 examples/scripts/rloo/rloo_GSM8K.py \
    --learning_rate 3e-6 \
    --output_dir models/GSM8K/ppo \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --total_episodes 30000 \
    --model_name_or_path microsoft/rho-math-1b-v0.1 \
    --sft_model_path realtreetune/rho-1b-sft-GSM8K \
    --non_eos_penalty \
    --stop_token eos \
    --response_length 53 \
    --sanity_check

accelerate launch --config_file examples/accelerate_configs/deepspeed_zero2.yaml \
    examples/scripts/rloo/rloo_tldr.py \
    --output_dir models/minimal/rloo_tldr \
    --num_ppo_epochs 1 \
    --num_mini_batches 1 \
    --learning_rate 3e-6 \
    --per_device_train_batch_size 16 \
    --gradient_accumulation_steps 4 \
    --total_episodes 1000000 \
    --model_name_or_path EleutherAI/pythia-1b-deduped \
    --sft_model_path cleanrl/EleutherAI_pythia-1b-deduped__sft__tldr \
    --reward_model_path cleanrl/EleutherAI_pythia-1b-deduped__reward__tldr \
    --local_rollout_forward_batch_size 16 \
    --non_eos_penalty \
    --stop_token eos \
"""


if __name__ == "__main__":
    #wandb.init(project='trl', entity='swish')
    parser = HfArgumentParser((RLOOConfig, ModelConfig))
    config, model_config = parser.parse_args_into_dataclasses()
    # remove output_dir if exists
    shutil.rmtree(config.output_dir, ignore_errors=True)

    ################
    # Model & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        model_config.model_name_or_path,
        padding_side="left",
        trust_remote_code=model_config.trust_remote_code,
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_QUERY_CHAT_TEMPLATE
    # reward_model = AutoModelForSequenceClassification.from_pretrained(
    #     config.reward_model_path, trust_remote_code=model_config.trust_remote_code, num_labels=1
    # )
    ref_policy = AutoModelForCausalLM.from_pretrained(
        config.sft_model_path, trust_remote_code=model_config.trust_remote_code
    )
    policy = AutoModelForCausalLM.from_pretrained(
        config.sft_model_path, trust_remote_code=model_config.trust_remote_code
    )
    ################
    # Dataset
    ################
    cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
    raw_datasets = load_dataset("openai/gsm8k", 'main', cache_dir=cache_dir)
    #raw_datasets = load_dataset("trl-internal-testing/tldr-preference-sft-trl-style", cache_dir=cache_dir)

    if config.sanity_check:
        for key in raw_datasets:
            raw_datasets[key] = raw_datasets[key].select(range(1000))
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]
    
    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            example = {
                'content': element["question"],
                'role': 'user'
            }
            input_ids = tokenizer.apply_chat_template(
                #element["messages"][:1],
                [example],
                padding=False,
                add_generation_prompt=True,
            )
            # match = re.search(r'#### (\d+)', element["answer"])
            # # If a match is found, extract the number
            # if match:
            #     number = match.group(1)  # group(1) gives the part inside the parentheses
            # else:
            #     print("No match found")
            number = element["answer"].split('####')[1].strip()
            number_ex = {
                'content': number,
                'role': 'user'
            }
            response_ids = tokenizer.apply_chat_template(
                [number_ex],
                padding=False,
                add_generation_prompt=False,
            )
            response_ids.append(0)
            if len(response_ids) < config.response_length:
                response_ids += [tokenizer.pad_token_id] * (config.response_length - len(response_ids))
            
            return {"input_ids": input_ids, 
                    "lengths": len(input_ids), 
                    "response_ids":response_ids,
                    "response_lengths": len(response_ids)}

        return dataset.map(
            tokenize,
            remove_columns=dataset.column_names,
            num_proc=1 if config.sanity_check else multiprocessing.cpu_count(),
            load_from_cache_file=not config.sanity_check,
        )
    
    train_dataset = prepare_dataset(train_dataset, tokenizer)
    eval_dataset = prepare_dataset(eval_dataset, tokenizer)
    # filtering -> not sure how to define this tokens.
    train_dataset = train_dataset.filter(lambda x: x["lengths"] <= 512)
    eval_dataset = eval_dataset.filter(lambda x: x["lengths"] <= 512)
    train_dataset = train_dataset.filter(lambda x: x["response_lengths"] <= config.response_length)
    eval_dataset = eval_dataset.filter(lambda x: x["response_lengths"] <= config.response_length)

    assert train_dataset[0]["input_ids"][-1] != tokenizer.eos_token_id, "The last token should not be an EOS token"

    ################
    # Training
    ################
    # Reward model removed!
    trainer = RLOOTrainerReasoning(
        config=config,
        tokenizer=tokenizer,
        policy=policy,
        ref_policy=ref_policy,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )
    trainer.train()
    trainer.save_model(config.output_dir)
    if config.push_to_hub:
        trainer.push_to_hub()
    trainer.generate_completions()
