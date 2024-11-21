from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import AutoConfig

#huggingface-cli login
#huggingface-cli whoami

# Define the local folder containing the model files
local_folder = "/home/mila/j/johan.ceron/scratch/gfn_llm/models/GSM8K/ppo/"
config = AutoConfig.from_pretrained(local_folder)
print(config)

# Load the model and tokenizer from the local folder
model = AutoModelForCausalLM.from_pretrained(local_folder, trust_remote_code=False)
tokenizer = AutoTokenizer.from_pretrained(local_folder)
# print(f"Tokenizer vocab size: {tokenizer.vocab_size}")
# if tokenizer.pad_token_id is None:
#     tokenizer.add_special_tokens({"pad_token": "[PAD]"})
# if tokenizer.eos_token_id is None:
#     tokenizer.add_special_tokens({"eos_token": "<eos>"})

# # Align padding tokens between tokenizer and model
# model.config.pad_token_id = tokenizer.pad_token_id
# model.config.eos_token_id = tokenizer.eos_token_id

# Push the model and tokenizer to the Hugging Face Hub
model.push_to_hub("rlooG")
tokenizer.push_to_hub("rlooG")
