#!/bin/bash

# Use the variable for the job name and log/error files
#$ -N subpy450-410
#$ -o /exports/eddie/scratch/s2558433/job_runs/SUBPY-$JOB_ID.log
#$ -e /exports/eddie/scratch/s2558433/job_runs/SUBPY-$JOB_ID.err
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

conda activate pythia

# python main.py --N 10000 --batch-size 10 --model1 EleutherAI/pythia-2.8b --model2 EleutherAI/pythia-2.8b --corpus-path "ArmelR/the-pile-splitted" --corpus-subset "ArXiv" --name-tag "10karxiv" --is-splitted

# python main.py --N 10000 --batch-size 10 --model1 EleutherAI/pythia-2.8b --model2 EleutherAI/pythia-2.8b --corpus-path "ArmelR/the-pile-splitted" --corpus-subset "Enron Emails" --name-tag "10kenronemails" --is-splitted

# python main.py --N 10000 --batch-size 10 --model1 EleutherAI/pythia-2.8b --model2 EleutherAI/pythia-2.8b --corpus-path "ArmelR/the-pile-splitted" --corpus-subset "Gutenberg (PG-19)" --name-tag "10kgutenberg"--is-splitted

python main.py --N 10000 --batch-size 10 --model1 EleutherAI/pythia-2.8b --model2 EleutherAI/pythia-2.8b --corpus-path "ArmelR/the-pile-splitted" --corpus-subset "Github" --name-tag "10kgit-input-450" --is-splitted --input-len 450

python main.py --N 10000 --batch-size 10 --model1 EleutherAI/pythia-2.8b --model2 EleutherAI/pythia-2.8b --corpus-path "ArmelR/the-pile-splitted" --corpus-subset "Wikipedia (en)" --name-tag "10kwiki-input-450" --is-splitted --input-len 450

# python main.py --N 10000 --batch-size 10 --model1 EleutherAI/pythia-2.8b --model2 EleutherAI/pythia-2.8b --corpus-path "ArmelR/the-pile-splitted" --corpus-subset "DM Mathematics" --name-tag "10kDM"--is-splitted  --input-len 450


conda deactivate 
