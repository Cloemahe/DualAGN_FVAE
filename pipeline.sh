#!/bin/sh
#SBATCH --job-name=python_VAE_zscale
#SBATCH --time=18:00:00
#SBATCH --account=supermassifs
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --output=output.log

echo Begin

conda init
conda activate dual_agn_fvae_env

python -u ./scripts/flow_vae.py --config ./config/flow_vae.yml

echo End
conda deactivate
