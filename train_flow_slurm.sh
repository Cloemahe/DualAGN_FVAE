#!/bin/sh
#SBATCH --job-name=python_VAE_zscale
#SBATCH --time=01:00:00
#SBATCH --account=supermassifs
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --output=output_flow_1.log

echo Begin

conda init
conda activate dual_agn_fvae_env

python -u ./scripts/train_flow.py --config ./config/train_flow.yml

echo End
conda deactivate
