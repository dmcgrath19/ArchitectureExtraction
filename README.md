# Master’s Dissertation Experiments: Training Data Memorization in LLMs

This repository contains code for the prefix experiments conducted as the first part of my master’s dissertation, comparing training data memorization between different deep learning architectures—specifically SSMs (Structured State Space Models) and transformers. Mamba and Pythia models were chosen as representatives from the respective architectures due to size and training comparability.

The experiments are based on the work of [Carlini](https://github.com/ftramer/LM_Memorization), aiming to extend their analysis by examining factors such as model size, input length, and data type across architectures. The objective is to assess how these factors affect memorization trends across architectures.

## Overview

### Experiments Conducted
- **Comparison of Model Sizes**: Different model sizes were used to explore the relationship between model capacity and memorization.
- **Input Length Variations**: We tested different input lengths to understand the effect of context length on memorization behavior.
- **Data Type**: The experiments utilized subsets from *The Pile* dataset, comparing how different data types impact memorization rates.

The code in this repository is designed to gather outputs from models and store them in CSV format for subsequent analysis. The collected data was further processed to evaluate and compare the memorization characteristics of each model.

## Contents
- `scripts/`: Scripts used to train models and collect output data.
- `setup.md`: Detailed setup instructions, including the techniques used to configure and run the scripts.
- `model_utils.py`: Streams and parses data from hf to be used as samples, calculates perplexity & prints samples from evaluation
- `main.py`: Loads the models, performs the prefix attack, & stores the response and output per model in a corresponding csv

## Getting Started

To set up the environment and replicate the experiments, please refer to [setup.md](setup.md). This document provides step-by-step instructions and describes the techniques used to configure and execute the scripts.

### Requirements
- Python 3.9>
- Required libraries: [list libraries here or provide details in `setup.md`]
- Model checkpoints: Download links or paths (as mentioned in `setup.md`)

### Running the Scripts
1. **Setup Environment**: Follow the instructions in [setup.md](setup.md) to set up the required environment.
2. **Run Experiments**: Use the scripts in `scripts/` to run model comparisons.


## Acknowledgments
This work builds upon the foundational research of Carlini et al., whose contributions to understanding memorization in LLMs served as a basis for these experiments.
