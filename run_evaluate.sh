#!/bin/bash
#SBATCH --time=3:00:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:a100l:1
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

python3 trl/trainer/evaluate.py \
    --output_dir models/GSM8K/ppo \
    --sft_model_path realtreetune/rho-1b-sft-GSM8K \
    --sanity_check