#!/bin/bash

# Define the job name variable
JOB_NAME="mamba-130m"

# Use the variable for the job name and log/error files
#$ -N mamba-160m
#$ -o /exports/eddie/scratch/s2558433/mamba-130m_$JOB_ID.log
#$ -e /exports/eddie/scratch/s2558433/mamba-130m_$JOB_ID.err
#$ -cwd
#$ -pe sharedmem 16
#$ -l h_vmem=4G
#$ -l h_rt=48:00:00

# Create / activate conda env if it doesn't exist

export HF_HOME="/exports/eddie/scratch/s2558433/huggingface_cache"
export TRANSFORMERS_CACHE="/exports/eddie/scratch/s2558433/huggingface_cache/transformers"
export HF_DATASETS_CACHE="/exports/eddie/scratch/s2558433/huggingface_cache/datasets"
export PIP_CACHE_DIR="/exports/eddie/scratch/s2558433/.cache/pip"
export CONDA_PKGS_DIRS="/exports/eddie/scratch/s2558433/.cache/conda_pkgs"
export HF_DATASETS_CACHE="/exports/eddie/scratch/s2558433/.cache/huggingface_datasets"

source /exports/eddie/scratch/s2558433/miniconda3/etc/profile.d/conda.sh

cd /exports/eddie/scratch/s2558433/ArchitectureExtraction/
#conda remove --name extract --all

conda activate mamba

pip install -r requirements.txt

# Run the main script
python main.py --N 1 --batch-size 10 --model1 state-spaces/mamba-2.8b-hf --model2 state-spaces/mamba-130m-hf --corpus-path monology/pile-uncopyrighted
#python main.py --N 1000 --batch-size 10 --model1 EleutherAI/pythia-2.8b --model2 EleutherAI/pythia-410m --cor

conda deactivate 
