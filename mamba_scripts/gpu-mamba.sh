#!/bin/bash

# Use the variable for the job name and log/error files
#$ -N CHAI-KING
#$ -o /exports/eddie/scratch/s2558433/job_runs/CHAI-mambax-130m_$JOB_ID.log
#$ -e /exports/eddie/scratch/s2558433/job_runs/CHAI-mambax-130m_$JOB_ID.err
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

export CXXFLAGS="-std=c99"
export CFLAGS="-std=c99"
export TOKENIZERS_PARALLELISM=false

. /etc/profile.d/modules.sh
module unload cuda

module load cuda/12.1.1
#qlogin -q gpu -pe gpu-a100 1 -l h_vmem=500G -l h_rt=24:00:00

source /exports/eddie/scratch/s2558433/miniconda3/etc/profile.d/conda.sh
module load anaconda

cd /exports/eddie/scratch/s2558433/ArchitectureExtraction/
conda activate mamba
#pip install causal-conv1d>=1.2.0
#pip install mamba-ssm

#pip install -r requirements.txt

# Run the main script
python main.py --N 10 --batch-size 10 --model1 state-spaces/mamba-2.8b-hf --model2 state-spaces/mamba-130m-hf --corpus-path monology/pile-uncopyrighted --name-tag king-CHAI --corpus-subset 

conda deactivate


