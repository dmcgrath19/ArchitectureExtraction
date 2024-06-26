#!/bin/bash

# Use the variable for the job name and log/error files
#$ -N py-1.4b-mono
#$ -o /exports/eddie/scratch/s2558433/job_runs/PI-mono-1.4b_$JOB_ID.log
#$ -e /exports/eddie/scratch/s2558433/job_runs/PI-mono-1.4b_$JOB_ID.err
#$ -cwd
#$ -q gpu
#$ -pe gpu-a100 1
#$ -l h_vmem=500G
#$ -l h_rt=24:00:00
#$ -m bea -M s2558433@ed.ac.uk 

export HF_HOME="/exports/eddie/scratch/s2558433/.cache/huggingface_cache"
export TRANSFORMERS_CACHE="/exports/eddie/scratch/s2558433/.cache/huggingface_cache/transformers"
export HF_DATASETS_CACHE="/exports/eddie/scratch/s2558433/.cache/huggingface_cache/datasets"
export PIP_CACHE_DIR="/exports/eddie/scratch/s2558433/.cache/pip"
export CONDA_PKGS_DIRS="/exports/eddie/scratch/s2558433/.cache/conda_pkgs"

source /exports/eddie/scratch/s2558433/miniconda3/etc/profile.d/conda.sh

cd /exports/eddie/scratch/s2558433/ArchitectureExtraction/
#conda remove --name extract --all


conda activate pythia


#pip install -r requirements.txt

# Run the main script
python pythia-main.py --N 10000 --batch-size 10 --model1 EleutherAI/pythia-2.8b --model2 EleutherAI/pythia-1.4b --corpus-path "monology/pile-uncopyrighted" --name-tag 10k-pile-1 
#python main.py --N 1000 --batch-size 10 --model1 EleutherAI/pythia-2.8b --model2 EleutherAI/pythia-410m --cor
# python pythia-main.py --N 10 --batch-size 10 --model1 EleutherAI/pythia-2.8b --model2 EleutherAI/pythia-160m --corpus-path "ArmelR/the-pile-splitted" --corpus-subset "DM Mathematics"
conda deactivate 
