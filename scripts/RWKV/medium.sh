#!/bin/bash

# Use the variable for the job name and log/error files
#$ -N rkvw-XL
#$ -o /exports/eddie/scratch/s2558433/job_runs/rwkv-$JOB_ID.log
#$ -e /exports/eddie/scratch/s2558433/job_runs/rwkv-$JOB_ID.err
#$ -cwd
#$ -l rl9=true
#$ -q gpu
#$ -pe gpu-a100 1
#$ -l h_vmem=300G
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


source /exports/eddie/scratch/s2558433/miniconda3/etc/profile.d/conda.sh
module load anaconda

cd /exports/eddie/scratch/s2558433/ArchitectureExtraction/

conda activate rwkv


python main.py --N 6000 --batch-size 10 --model1 RWKV/rwkv-4-3b-pile --model2 RWKV/rwkv-4-3b-pile --corpus-path "monology/pile-uncopyrighted" --name-tag "10kpile-a"  
python main.py --N 3000 --batch-size 10 --model1 RWKV/rwkv-4-3b-pile --model2 RWKV/rwkv-4-3b-pile --corpus-path "monology/pile-uncopyrighted" --name-tag "8kpile-b"  
python main.py --N 2000 --batch-size 10 --model1 RWKV/rwkv-4-3b-pile --model2 RWKV/rwkv-4-3b-pile --corpus-path "monology/pile-uncopyrighted" --name-tag "2kpile-c"  
python main.py --N 2000 --batch-size 10 --model1 RWKV/rwkv-4-3b-pile --model2 RWKV/rwkv-4-3b-pile --corpus-path "monology/pile-uncopyrighted" --name-tag "2kpile-d"  


conda deactivate 
