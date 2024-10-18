import multiprocessing
import shutil
import wandb
from tqdm import tqdm

from accelerate import Accelerator   

from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoModelForSequenceClassification,
    AutoTokenizer,
    HfArgumentParser,
)
import re
import os
from trl.trainer import ModelConfig #(Ask to SIDDARTH)
from trl.trainer.rloo_trainer_reasoning import RLOOConfig, RLOOTrainerReasoning

from trl.trainer.utils import SIMPLE_QUERY_CHAT_TEMPLATE
from trl.models.utils import unwrap_model_for_generation
from trl.trainer.utils import (
    generate,
)
from transformers import (
    GenerationConfig,
)
import torch

INVALID_LOGPROB = 1.0
FIND_NUMBERS_REGEX = re.compile(
    r"(?:[+-]?\d+\.\d*|[+-]?\.\d+|[+-]?\d+e[-+]?\d+|[+-]?\d+)"
)


"""
module unload anaconda
echo "loading modules"
module load python/3.10 cudatoolkit/12.3.2

echo "loading env"
cd $HOME/johan_phd/
source llm_gfn_git/bin/activate

echo "running script.."
cd $HOME/scratch/gfn_llm/

python3 trl/trainer/evaluate.py \
    --output_dir models/GSM8K/ppo \
    --sft_model_path realtreetune/rho-1b-sft-GSM8K \
    --sanity_check

"""

if __name__ == "__main__":
    parser = HfArgumentParser((RLOOConfig, ModelConfig))
    config, model_config = parser.parse_args_into_dataclasses()
    # remove output_dir if exists
    shutil.rmtree(config.output_dir, ignore_errors=True)

    ################
    # Model & Tokenizer
    ################
    tokenizer = AutoTokenizer.from_pretrained(
        config.sft_model_path,
        padding_side="left",
        trust_remote_code=model_config.trust_remote_code,
    )
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    if tokenizer.chat_template is None:
        tokenizer.chat_template = SIMPLE_QUERY_CHAT_TEMPLATE
    policy = AutoModelForCausalLM.from_pretrained(
        config.sft_model_path, trust_remote_code=model_config.trust_remote_code
    )
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

    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            input_ids = tokenizer(
                element["question"],
            )
            text_answer = element["answer"]
            number = parse_number(element["answer"].split('####')[1])
            return {"input_ids": input_ids['input_ids'], 
                    "lengths": len(input_ids['input_ids']), 
                    "response_ids":number,
                    "text_answer":text_answer,
                    "raw_question":element["question"]}

        return dataset.map(
            tokenize,
            remove_columns=dataset.column_names,
            num_proc=1 if config.sanity_check else multiprocessing.cpu_count(),
            load_from_cache_file=not config.sanity_check,
        )
    
    train_dataset = prepare_dataset(train_dataset, tokenizer)
    eval_dataset = prepare_dataset(eval_dataset, tokenizer)
    
    def grade_answer(given_answer, ground_truth):
        if given_answer is None:
            return torch.tensor(0.0)
        given_answer = torch.tensor(float(given_answer))
        assert ground_truth is not None
        comparison = torch.isclose(given_answer, ground_truth, atol=1e-5)
        return comparison.view(-1, 1)  # Reshape to (64, 1)
    
    def extract_predicted_answer_from_text(text: str, use_original_format:bool=False):
        if use_original_format:
            # Extract the final answer based on ####
            if "####" not in text:
                return None
            parts = text.split("####")
            assert len(parts) >= 2
            return parts[-1].strip()

        text = text.replace(",", "")
        pred_answer = FIND_NUMBERS_REGEX.findall(text)  # TODO: add task to attributes
        if len(pred_answer) == 0:
            return None
        else:
            # Pick the last number
            pred_answer = pred_answer[-1].strip()
            return pred_answer
    
    accelerator = Accelerator(gradient_accumulation_steps=4)
    device = accelerator.device

    type = 'eval'
    if type == 'eval':
        print('===Eval Dataset:', type)
        queries = eval_dataset["input_ids"]
        ground_truth_data = eval_dataset["response_ids"] # -> len(ground_truth) 1319
        ground_truth_text = eval_dataset["text_answer"]
        raw_question = eval_dataset["raw_question"]
    else:
        print('===Training Dataset:', type)
        queries = train_dataset["input_ids"]
        ground_truth_data = train_dataset["response_ids"] # -> len(ground_truth) 1319
        ground_truth_text = train_dataset["text_answer"]
        raw_question = eval_dataset["raw_question"]
   
    response_length = 53
    temperature = 0.7
    generation_config = GenerationConfig(
            max_new_tokens=response_length,
            min_new_tokens=response_length,
            temperature=(temperature + 1e-7),
            top_k=0.0,
            top_p=1.0,
            do_sample=True,
        )
    
    # Initialize a list to store the scores
    scores = []
    use_original_format = False
    print("===use_original_format:\n", use_original_format)
    with unwrap_model_for_generation(policy, accelerator) as unwrapped_model:
        for i in tqdm(range(0, len(queries))):
            query = queries[i] #->74, this value vary.
            #print("===1. query: ", query)
            query = torch.tensor(query).unsqueeze(0)
            #print("===2. query: ", query)
            context_length = query.shape[1]
            #print("===3. context_length: ", context_length)
            ground_truth = ground_truth_data[i]
            #print("===4. ground_truth: ", ground_truth)
            raw_question = eval_dataset["raw_question"][i]
            #print("===5. raw_question: ", raw_question)
            raw_question_decode = tokenizer.batch_decode(query, skip_special_tokens=True)
            #print("===6. raw_question_decode: ", raw_question_decode[0])
            query_response, logits = generate(
                unwrapped_model,
                query,
                tokenizer.pad_token_id,
                generation_config,
            )
            #print("===7. query_response: ", query_response)            
            pred_answer = query_response[:, context_length:]
            #print("===8. pred_answer: ", pred_answer) 
            pred_answer = tokenizer.batch_decode(pred_answer, skip_special_tokens=True)
            print("===9. pred_answer: ", pred_answer) 
            pred_answer = extract_predicted_answer_from_text(text=pred_answer[0], 
                                                            use_original_format=use_original_format)
            print("===10. pred_answer: ", pred_answer)    
            #print("===11. ground_truth_text: ", ground_truth_text[i])
            print("===12. ground_truth: ", torch.tensor(ground_truth))
            score = grade_answer(pred_answer, torch.tensor(ground_truth)) # binary_RM
            #print("===13. score: ", score) 
            #print("===14. score.item(): ", score.item()) 
            scores.append(score.item())
            #print("===15. score: ", scores) 
             # Calculate the average score
            average_score = sum(scores) / len(scores)
            print(f"Average Score: {average_score}")
print(f" Final Average Score: {average_score}")

