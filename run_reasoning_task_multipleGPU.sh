#!/bin/bash
#SBATCH --time=100:00:00
#SBATCH --mem=80G
#SBATCH --partition lab-bengioy
#SBATCH --gres=gpu:a100l:3
#SBATCH --cpus-per-task=4
#SBATCH --mail-user=jobando0730@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --array=1

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
    examples/scripts/rloo/rloo_GSM8K_vllm.py \
    --learning_rate 3e-6 \
    --output_dir models/GSM8K/ppo \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 64 \
    --total_episodes 50000 \
    --model_name_or_path microsoft/rho-math-1b-v0.1 \
    --sft_model_path realtreetune/rho-1b-sft-GSM8K \
    --non_eos_penalty \
    --stop_token eos \
    --response_length 512 \
    --sanity_check