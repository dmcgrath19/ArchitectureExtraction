#!/bin/sh

#SBATCH --partition=PGR-Standard
#SBATCH --mem=12000
#SBATCH --gres=gpu:1

python mlp-main.py --N 10000 --batch-size 10 --model1 models/pythia-1.4b --model2 models/pythia-160m --corpus-path "pile.txt" --is-local --name-tag "450step" --input-len 450

python mlp-main.py --N 10000 --batch-size 10 --model1 models/pythia-1.4b --model2 models/pythia-160m --corpus-path "pile.txt" --is-local --name-tag "50step" --input-len 50

# python mlp-main.py --N 10000 --batch-size 10 --model1 models/pythia-1.4b --model2 models/pythia-160m --corpus-path "pile.txt" --is-local --name-tag "900step" --input-len 900


#50
#900