#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Teach-Standard
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-08:00:00

# export CUDA_HOME=/opt/cuda-9.0.176.1/
# export CUDNN_HOME=/opt/cuDNN-7.0/
export STUDENT_ID=$(whoami)
# export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
# export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH
# export CPATH=${CUDNN_HOME}/include:$CPATH
# export PATH=${CUDA_HOME}/bin:${PATH}
# export PYTHON_PATH=$PATH


. /etc/profile.d/modules.sh
module unload cuda
module load cuda/12.1.1

export HF_HOME="home/s2558433/.cache/huggingface_cache"
export TRANSFORMERS_CACHE="home/s2558433/.cache/huggingface_cache/transformers"
export HF_DATASETS_CACHE="home/s2558433/.cache/huggingface_cache/datasets"
export PIP_CACHE_DIR="home/s2558433/.cache/pip"
export CONDA_PKGS_DIRS="home/s2558433/.cache/conda_pkgs"

export CXXFLAGS="-std=c99"
export CFLAGS="-std=c99"
export TOKENIZERS_PARALLELISM=false

which nvcc

source /home/${STUDENT_ID}/miniconda3/bin/activate mambafour
cd /home/s2558433/ArchitectureExtraction/local-env/

pip install -r mamba_requirements.txt

