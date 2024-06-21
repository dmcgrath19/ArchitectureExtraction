#!/bin/bash

# Define the job name variable
JOB_NAME="pythia-160m"

# Use the variable for the job name and log/error files
#$ -N pythia-160m
#$ -o /exports/eddie/scratch/s2558433/job_runs/pythia-160m_$JOB_ID.log
#$ -e /exports/eddie/scratch/s2558433/job_runs/pythia-160m_$JOB_ID.err
#$ -cwd
#$ -pe sharedmem 16
#$ -l h_vmem=4G
#$ -l h_rt=48:00:00

export HF_HOME="/exports/eddie/scratch/s2558433/.cache/huggingface_cache"
export TRANSFORMERS_CACHE="/exports/eddie/scratch/s2558433/.cache/huggingface_cache/transformers"
export HF_DATASETS_CACHE="/exports/eddie/scratch/s2558433/.cache/huggingface_cache/datasets"
export PIP_CACHE_DIR="/exports/eddie/scratch/s2558433/.cache/pip"
export CONDA_PKGS_DIRS="/exports/eddie/scratch/s2558433/.cache/conda_pkgs"

source /exports/eddie/scratch/s2558433/miniconda3/etc/profile.d/conda.sh

cd /exports/eddie/scratch/s2558433/
#conda remove --name extract --all

#conda create -n extracted-1b python=3.9 
conda create -n extracted python=3.9 

conda activate extracted-1b

cd ArchitectureExtraction

pip install -r requirements.txt

# Run the main script
python main.py --N 1000 --batch-size 10 --model1 EleutherAI/pythia-2.8b --model2 EleutherAI/pythia-160m --corpus-path monology/pile-uncopyrighted
#python main.py --N 1000 --batch-size 10 --model1 EleutherAI/pythia-2.8b --model2 EleutherAI/pythia-410m --cor

conda deactivate 