#!/bin/bash

#$ -N cpu-mamba-160mm
#$ -o /exports/eddie/scratch/s2558433/job_runs/cpu-mamba-ex-160m_$JOB_ID.log
#$ -e /exports/eddie/scratch/s2558433/job_runs/cpu-mamba-ex-160m_$JOB_ID.err
#$ -cwd
#$ -pe sharedmem 16
#$ -l h_vmem=8G
#$ -l h_rt=48:00:00
#$ -m bea -M s2558433@ed.ac.uk 

export HF_HOME="/exports/eddie/scratch/s2558433/.cache/huggingface_cache"
export TRANSFORMERS_CACHE="/exports/eddie/scratch/s2558433/.cache/huggingface_cache/transformers"
export HF_DATASETS_CACHE="/exports/eddie/scratch/s2558433/.cache/huggingface_cache/datasets"
export PIP_CACHE_DIR="/exports/eddie/scratch/s2558433/.cache/pip"
export CONDA_PKGS_DIRS="/exports/eddie/scratch/s2558433/.cache/conda_pkgs"

source /exports/eddie/scratch/s2558433/miniconda3/etc/profile.d/conda.sh

cd /exports/eddie/scratch/s2558433/

conda activate extracted

cd ArchitectureExtraction

#python main.py --N 1000 --batch-size 10 --model1 state-spaces/mamba-2.8b-hf --model2 state-spaces/mamba-130m-hf --corpus-path monology/pile-uncopyrighted
python main.py --N 10 --batch-size 10 --model1 state-spaces/mamba-2.8b-hf --model2 state-spaces/mamba-370m-hf --corpus-path "ArmelR/the-pile-splitted" --corpus-subset "DM Mathematics" --name-tag '10k-dm-math' 


conda deactivate