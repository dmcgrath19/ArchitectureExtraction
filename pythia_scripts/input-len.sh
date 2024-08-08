#!/bin/bash

# Use the variable for the job name and log/error files
#$ -N inputLlen-Pythia
#$ -o /exports/eddie/scratch/s2558433/job_runs/py-input-$JOB_ID.log
#$ -e /exports/eddie/scratch/s2558433/job_runs/py-input-$JOB_ID.err
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

#qlogin -q gpu -pe gpu-a100 1 -l h_vmem=500G -l h_rt=24:00:00

source /exports/eddie/scratch/s2558433/miniconda3/etc/profile.d/conda.sh
module load anaconda

cd /exports/eddie/scratch/s2558433/ArchitectureExtraction/

conda activate pythia



# python main.py --N 10000 --batch-size 10 --model1 EleutherAI/pythia-2.8b --model2 EleutherAI/pythia-410m --corpus-path "monology/pile-uncopyrighted" --name-tag "10kpile"  

python main.py --N 10000 --batch-size 10 --model1 EleutherAI/pythia-1b --model2 EleutherAI/pythia-1b --corpus-path "monology/pile-uncopyrighted" --name-tag "10kpile" 

python main.py --N 10000 --batch-size 10 --model1 EleutherAI/pythia-160m --model2 EleutherAI/pythia-160m --corpus-path "monology/pile-uncopyrighted" --name-tag "10kpile" 

# python main.py --N 10000 --batch-size 10 --model1 EleutherAI/pythia-2.8b --model2 EleutherAI/pythia-2.8b --corpus-path "monology/pile-uncopyrighted" --name-tag "10kpile" 

# python main.py --N 10000 --batch-size 10 --model1 EleutherAI/pythia-2.8b --model2 EleutherAI/pythia-1.4b --corpus-path "monology/pile-uncopyrighted" --name-tag "10kpile" 





conda deactivate 
