#!/bin/bash

# Use the variable for the job name and log/error files
#$ -N gptneox-a
#$ -o /exports/eddie/scratch/s2558433/job_runs/py-input-$JOB_ID.log
#$ -e /exports/eddie/scratch/s2558433/job_runs/py-input-$JOB_ID.err
#$ -cwd
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


python main.py --N 10000 --batch-size 10 --model1 EleutherAI/gpt-neo-125m --model2 EleutherAI/gpt-neo-125m --corpus-path "monology/pile-uncopyrighted" --name-tag "10kpileq"  


python main.py --N 10000 --batch-size 10 --model1 EleutherAI/gpt-neo-2.7B --model2 EleutherAI/gpt-neo-125m --corpus-path "monology/pile-uncopyrighted" --name-tag "10kpileqa" 

python main.py --N 10000 --batch-size 10 --model1 EleutherAI/gpt-neo-2.7B --model2 EleutherAI/gpt-neo-125m --corpus-path "monology/pile-uncopyrighted" --name-tag "10kpileqb" 

python main.py --N 10000 --batch-size 10 --model1 EleutherAI/gpt-neo-2.7B --model2 EleutherAI/gpt-neo-125m --corpus-path "monology/pile-uncopyrighted" --name-tag "10kpileqc" 

python main.py --N 10000 --batch-size 10 --model1 EleutherAI/gpt-neo-2.7B --model2 EleutherAI/gpt-neo-2.7B --corpus-path "monology/pile-uncopyrighted" --name-tag "10kpileg" 

python main.py --N 10000 --batch-size 10 --model1 EleutherAI/gpt-neo-2.7B --model2 EleutherAI/gpt-neo-1.3B --corpus-path "monology/pile-uncopyrighted" --name-tag "10kpileg" 

conda deactivate 
