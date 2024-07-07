#!/bin/sh
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --partition=Teach-Standard
#SBATCH --gres=gpu:1
#SBATCH --mem=12000  # memory in Mb
#SBATCH --time=0-08:00:00
#SBATCH -e home/s2558433/job_errors_$JOB_ID.log  # error file for stderr
#SBATCH -o home/s2558433/job_output_$JOB_ID.log  # output file for stdout

# export CUDA_HOME=/opt/cuda-9.0.176.1/
# export CUDNN_HOME=/opt/cuDNN-7.0/
export STUDENT_ID=$(whoami)
# export LD_LIBRARY_PATH=${CUDNN_HOME}/lib64:${CUDA_HOME}/lib64:$LD_LIBRARY_PATH
# export LIBRARY_PATH=${CUDNN_HOME}/lib64:$LIBRARY_PATH
# export CPATH=${CUDNN_HOME}/include:$CPATH
# export PATH=${CUDA_HOME}/bin:${PATH}
# export PYTHON_PATH=$PATH

echo "Loading module environment..."
. /etc/profile.d/modules.sh

# List all available modules and capture output to job_output.out
echo "Listing all available modules..."
module avail >> job_output.out

# Check currently loaded modules and capture output to job_output.out
echo "Checking currently loaded modules..."
module list >> job_output.out


module avail cuda


# Unload any existing CUDA module and load the desired version
module unload cuda
module load cuda/12.1.1

# Check if the CUDA module was loaded correctly
echo "Checking if CUDA module is loaded..."
module list

# Check if nvcc is available
echo "Checking if nvcc is available in PATH..."
which nvcc

# If nvcc is not found, exit the script with an error
if [ $? -ne 0 ]; then
    echo "Error: nvcc not found. CUDA module may not be loaded correctly."
    exit 1
fi

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

