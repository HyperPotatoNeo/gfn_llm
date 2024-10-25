import multiprocessing
import shutil
from tqdm import tqdm
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    HfArgumentParser,
)
import re
import os
from trl.trainer import ModelConfig #(Ask to SIDDARTH)
from trl.trainer.rloo_trainer_reasoning import RLOOConfig, RLOOTrainerReasoning
from vllm import LLM, SamplingParams
import torch

INVALID_LOGPROB = 1.0
FIND_NUMBERS_REGEX = re.compile(
    r"(?:[+-]?\d+\.\d*|[+-]?\.\d+|[+-]?\d+e[-+]?\d+|[+-]?\d+)"
)


"""
salloc --gres=gpu:a100l:1 --cpus-per-gpu=4 --mem=32G -t 10:00:00 --partition=unkillable
salloc --gres=gpu:a100l:2 -c 24 --mem=32G -t 12:00:00 --partition=lab-bengioy

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

python3 trl/trainer/evaluate.py \
    --output_dir models/GSM8K/ppo \
    --sft_model_path realtreetune/deepseekmath-7b-sft-GSM8K \
    --sanity_check

python3 trl/trainer/evaluate.py \
    --output_dir models/GSM8K/ppo \
    --sft_model_path realtreetune/rho-1b-sft-MATH \
    --sanity_check

python3 trl/trainer/evaluate.py \
    --output_dir models/GSM8K/ppo \
    --sft_model_path realtreetune/deepseekmath-7b-sft-MATH-v2 \
    --sanity_check
"""

if __name__ == "__main__":
    parser = HfArgumentParser((RLOOConfig, ModelConfig))
    config, model_config = parser.parse_args_into_dataclasses()
    shutil.rmtree(config.output_dir, ignore_errors=True)

    tokenizer = AutoTokenizer.from_pretrained(
        config.sft_model_path,
        #trust_remote_code=model_config.trust_remote_code,
    )
    #tokenizer.pad_token = tokenizer.eos_token
    # BOS and EOS tokens (check if your model defines them explicitly).
    bos_token = tokenizer.bos_token or "<bos>"
    eos_token = tokenizer.eos_token or "<eos>"

    llm = LLM(model=config.sft_model_path)

    cache_dir = os.path.expanduser("~/.cache/huggingface/datasets")
    raw_datasets = load_dataset("openai/gsm8k", 'main', cache_dir=cache_dir)
    train_dataset = raw_datasets["train"]
    eval_dataset = raw_datasets["test"]
    
    def parse_number(value):
        value = value.strip()
        value = value.replace(',', '')
        return float(value)
    
    def data_processing(query):
        question_template = f'{bos_token} [MATH_TASK] Problem: {query} Solution:'
        return question_template
    
    def prepare_dataset(dataset, tokenizer):
        """pre-tokenize the dataset before training; only collate during training"""

        def tokenize(element):
            data_pross = data_processing(element["question"])
            input_ids = tokenizer(
                data_pross,
                padding=False,
            )
            text_answer = element["answer"]
            number = parse_number(element["answer"].split('####')[1])
            return {"input_ids": input_ids['input_ids'], 
                    "lengths": len(input_ids['input_ids']), 
                    "response_ids":number,
                    "text_answer":text_answer,
                    "raw_question":element["question"],
                    "raw_question_template":data_pross}

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
            return torch.tensor(False)
        assert ground_truth is not None
        comparison = torch.isclose(torch.tensor(given_answer), torch.tensor(ground_truth), atol=1e-5)
        return  comparison
    
    def extract_predicted_answer_from_text(text: str, use_original_format:bool=False):
        if use_original_format:
            # Extract the final answer based on ####
            if "####" not in text:
                return None
            parts = text.split("####")
            assert len(parts) >= 2
            digit = parts[-1].strip()
            digit = digit.replace(",", "")
            return float(digit)

        text = text.replace(",", "")
        pred_answer = FIND_NUMBERS_REGEX.findall(text)  # TODO: add task to attributes
        if len(pred_answer) == 0:
            return None
        else:
            # Pick the last number
            pred_answer = pred_answer[-1].strip()
            return float(pred_answer)

    type = 'eval'
    if type == 'eval':
        print('===Eval Dataset:', type)
        queries = eval_dataset["input_ids"]
        ground_truth_data = eval_dataset["response_ids"] # -> len(ground_truth) 1319
        ground_truth_text = eval_dataset["text_answer"]
        raw_question = eval_dataset["raw_question"]
        raw_question_template = eval_dataset["raw_question_template"]
    else:
        print('===Training Dataset:', type)
        queries = train_dataset["input_ids"]
        ground_truth_data = train_dataset["response_ids"] # -> len(ground_truth) 1319
        ground_truth_text = train_dataset["text_answer"]
        raw_question = train_dataset["raw_question"]
        raw_question_template = train_dataset["raw_question_template"]
   
    use_llm = True
    response_length = 1024
    temperature = 0.35
    top_p = 0.9 #1
    use_original_format = False
    print("===use_llm:", use_llm)
    print("===response_length:", response_length)
    print("===temperature:", temperature)
    print("===temperature:", top_p)
    print("===use_original_format:", use_original_format)   
    sampling_params = SamplingParams(temperature=temperature, top_p=top_p, max_tokens=response_length, stop="\n\n\nProblem:")

    # Initialize a list to store the scores
    scores = []
    for i in tqdm(range(0, len(queries))):
        query = queries[i] #->74, this value vary.
        #print("===1. query: ", query)
        query = torch.tensor(query).unsqueeze(0)
        #print("===2. query: ", query)
        context_length = query.shape[1]
        #print("===3. context_length: ", context_length)
        ground_truth = ground_truth_data[i]
        #print("===4. ground_truth: ", ground_truth)
        raw_questions = raw_question[i]
        #print("===5. raw_questions: ", raw_questions)
        #raw_question_decode = tokenizer.decode(query[0], skip_special_tokens=True)
        raw_question_decode = tokenizer.decode(query[0], skip_special_tokens=False)
        #print("===6. raw_question_decode: ", raw_question_decode)
        query = raw_question_template[i]
        outputs = llm.generate(query, sampling_params)
        #pred_answer = outputs[0].outputs[0].text
        decoded_text = tokenizer.decode(outputs[0].outputs[0].token_ids, skip_special_tokens=False)
        #print("===9B. pred_answer: ", pred_answer)
        #print("===9C. decoded_text: ", decoded_text)
        pred_answer = extract_predicted_answer_from_text(text=decoded_text, 
                                                            use_original_format=use_original_format,
                                                            )
        print("===10. pred_answer: ", pred_answer)    
        #print("===11. ground_truth_text: ", ground_truth_text[i])
        print("===12. ground_truth: ", ground_truth)
        score = grade_answer(pred_answer, ground_truth) # binary_RM
        #print("===13. score: ", score) 
        print("===14. score.item(): ", score.item())
        scores.append(score.item())
        #print("===15. score: ", scores) 
        # Calculate the average score
        average_score = sum(scores) / len(scores)
        print(f"Average Score: {average_score}")
        #import pdb; pdb.set_trace()
        print('================================================================')
print(f" Final Average Score: {average_score}")

