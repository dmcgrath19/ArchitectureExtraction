#!/bin/bash

# Use the variable for the job name and log/error files
#$ -N PY-410
#$ -o /exports/eddie/scratch/s2558433/job_runs/Pythia410_$JOB_ID.log
#$ -e /exports/eddie/scratch/s2558433/job_runs/Pythia410_$JOB_ID.err
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

#conda create -n pythia python=3.9 

conda activate pythia

#pip install -r requirements.txt

# Run the main script

python main.py --N 10 --batch-size 10 --model1 EleutherAI/pythia-2.8b --model2 EleutherAI/pythia-410m --corpus-path "ArmelR/the-pile-splitted" --corpus-subset "DM Mathematics" --name-tag "problem-withDM"
# mv output*.txt output*.csv /exports/e≈ddie/scratch/s2558433/ArchitectureExtraction/prev-runs/pythia-410/

# python pythia-main.py --N 10000 --batch-size 10 --model1 EleutherAI/pythia-2.8b --model2 EleutherAI/pythia-410m --corpus-path "ArmelR/the-pile-splitted" --corpus-subset "Github" --name-tag "10k-git"
# # mv output*.txt output*.csv /exports/eddie/scratch/s2558433/ArchitectureExtraction/prev-runs/pythia-410/

# python pythia-main.py --N 10000 --batch-size 10 --model1 EleutherAI/pythia-2.8b --model2 EleutherAI/pythia-410m --corpus-path "ArmelR/the-pile-splitted" --corpus-subset "Wikipedia (en)" --name-tag "10k-wiki"
# # mv output*.txt output*.csv /exports/eddie/scratch/s2558433/ArchitectureExtraction/prev-runs/pythia-410/

conda deactivate 
