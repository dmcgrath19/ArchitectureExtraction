
#!/bin/bash

# Use the variable for the job name and log/error files
#$ -N base
#$ -o /exports/eddie/scratch/s2558433/job_runs/base$JOB_ID.log
#$ -e /exports/eddie/scratch/s2558433/job_runs/base$JOB_ID.err
#$ -cwd
#$ -q gpu
#$ -pe gpu-a100 1
#$ -l h_vmem=200G
#$ -l h_rt=12:00:00
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

#conda create -n mamba python=3.9
conda activate mamba
# #conda install nvidia::cuda-toolkit=12.1
# pip install causal-conv1d>=1.2.0
# pip install mamba-ssm


python main.py --N 10000 --batch-size 10 --model1 state-spaces/mamba-2.8b-hf --model2 state-spaces/mamba-370m-hf --corpus-path 'KaiNylund/WMT-year-splits' --split "2021_train" --name-tag 10k-base --is-mamba --is-wmt

python main.py --N 10000 --batch-size 10 --model1 EleutherAI/pythia-2.8b --model2 EleutherAI/pythia-410m --corpus-path 'KaiNylund/WMT-year-splits' --split "2021_train" --name-tag 10k-base --is-wmt

conda deactivate
