
#!/bin/bash

# Use the variable for the job name and log/error files
#$ -N sub-mamba
#$ -o /exports/eddie/scratch/s2558433/job_runs/submam_$JOB_ID.log
#$ -e /exports/eddie/scratch/s2558433/job_runs/submam_$JOB_ID.err
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

source /exports/eddie/scratch/s2558433/miniconda3/etc/profile.d/conda.sh

cd /exports/eddie/scratch/s2558433/ArchitectureExtraction/

conda activate mamba

python main.py --N 10000 --batch-size 10 --model1 state-spaces/mamba-2.8b-hf --model2 state-spaces/mamba-370m-hf --corpus-path "ArmelR/the-pile-splitted" --corpus-subset "DM Mathematics" --name-tag "10kDM" --is-mamba --is-splitted

python main.py --N 10000 --batch-size 10 --model1 state-spaces/mamba-2.8b-hf --model2 state-spaces/mamba-370m-hf --corpus-path "ArmelR/the-pile-splitted" --corpus-subset "Github" --name-tag "10kgit" --is-splitted

python main.py --N 10000 --batch-size 10 --model1 state-spaces/mamba-2.8b-hf --model2 state-spaces/mamba-370m-hf --corpus-path "ArmelR/the-pile-splitted" --corpus-subset "Wikipedia (en)" --name-tag "10kwiki" --is-mamba --is-splitted

conda deactivate 
