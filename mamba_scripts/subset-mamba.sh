
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

. /etc/profile.d/modules.sh
module unload cuda
module load cuda/12.1.1

source /exports/eddie/scratch/s2558433/miniconda3/etc/profile.d/conda.sh
module load anaconda


cd /exports/eddie/scratch/s2558433/ArchitectureExtraction/
conda activate mambafour

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

. /etc/profile.d/modules.sh
module unload cuda
module load cuda/12.1.1

source /exports/eddie/scratch/s2558433/miniconda3/etc/profile.d/conda.sh
module load anaconda


cd /exports/eddie/scratch/s2558433/ArchitectureExtraction/
conda activate mambafour

# python main.py --N 100 --batch-size 10 --model1 state-spaces/mamba-2.8b-hf --model2 state-spaces/mamba-370m-hf --corpus-path 'KaiNylund/WMT-year-splits' --split "2021_train" --name-tag 10k-base --is-mamba --is-wmt

# python main.py --N 10000 --batch-size 10 --model1 state-spaces/mamba-2.8b-hf --model2 state-spaces/mamba-130m-hf --corpus-path "ArmelR/the-pile-splitted" --corpus-subset "ArXiv" --is-mamba --name-tag "10karxiv-input-150" --is-splitted #--input-len 450

# python main.py --N 10000 --batch-size 10 --model1 state-spaces/mamba-2.8b-hf --model2 state-spaces/mamba-130m-hf --corpus-path "ArmelR/the-pile-splitted" --corpus-subset "FreeLaw" --name-tag "10kfree-input-150" --is-mamba --is-splitted #--input-len 450

# python main.py --N 10000 --batch-size 10 --model1 state-spaces/mamba-2.8b-hf --model2 state-spaces/mamba-130m-hf --corpus-path "ArmelR/the-pile-splitted" --corpus-subset "EuroParl" --name-tag "10keuro-input-150" --is-mamba --is-splitted #--input-len 450

# python main.py --N 10000 --batch-size 10 --model1 state-spaces/mamba-2.8b-hf --model2 state-spaces/mamba-130m-hf --corpus-path "ArmelR/the-pile-splitted" --corpus-subset "Pile-CC" --name-tag "10kpilecc-input-150" --is-mamba --is-splitted #--input-len 450

# python main.py --N 10000 --batch-size 10 --model1 state-spaces/mamba-2.8b-hf --model2 state-spaces/mamba-130m-hf --corpus-path "ArmelR/the-pile-splitted" --corpus-subset "UPSTO Backgrounds" --name-tag "10kupsto-input-150" --is-mamba --is-splitted #--input-len 450

# python main.py --N 10000 --batch-size 10 --model1 state-spaces/mamba-2.8b-hf --model2 state-spaces/mamba-130m-hf --corpus-path "ArmelR/the-pile-splitted" --corpus-subset "HackerNews" --name-tag "10khack-input-150" --is-mamba --is-splitted #--input-len 450

python main.py --N 10000 --batch-size 10 --model1 state-spaces/mamba-2.8b-hf --model2 state-spaces/mamba-130m-hf --corpus-path "ArmelR/the-pile-splitted" --corpus-subset "PubMed Central" --name-tag "10kpubc-input-150" --is-mamba --is-splitted #--input-len 450

python main.py --N 10000 --batch-size 10 --model1 state-spaces/mamba-2.8b-hf --model2 state-spaces/mamba-130m-hf --corpus-path "ArmelR/the-pile-splitted" --corpus-subset "Enron Emails" --name-tag "10kenron-input-150" --is-mamba --is-splitted #--input-len 450

python main.py --N 10000 --batch-size 10 --model1 state-spaces/mamba-2.8b-hf --model2 state-spaces/mamba-130m-hf --corpus-path "ArmelR/the-pile-splitted" --corpus-subset "Wikipedia (en)" --name-tag "10kenron-input-150" --is-mamba --is-splitted #--input-len 450

# python main.py --N 10000 --batch-size 10 --model1 EleutherAI/pythia-2.8b --model2 EleutherAI/pythia-160m --corpus-path "ArmelR/the-pile-splitted" --corpus-subset "Enron Emails" --name-tag "10kenron-input-150"

python main.py --N 10000 --batch-size 10 --model1 state-spaces/mamba-2.8b-hf --model2 state-spaces/mamba-130m-hf --corpus-path "ArmelR/the-pile-splitted" --corpus-subset "Gutenberg (PG-19)" --name-tag "10kgutenberg-input-150" --is-mamba --is-splitted #--input-len 450

python main.py --N 10000 --batch-size 10 --model1 state-spaces/mamba-2.8b-hf --model2 state-spaces/mamba-130m-hf --corpus-path "ArmelR/the-pile-splitted" --corpus-subset "DM Mathematics" --is-mamba --name-tag "10kDM-input-150" --is-splitted #--input-len 450

python main.py --N 10000 --batch-size 10 --model1 state-spaces/mamba-2.8b-hf --model2 state-spaces/mamba-130m-hf --corpus-path "ArmelR/the-pile-splitted" --corpus-subset "Github" --name-tag "10kgit-input-150" --is-mamba --is-splitted #--input-len 450



conda deactivate 



conda deactivate 
