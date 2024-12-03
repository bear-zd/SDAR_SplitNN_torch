#!/bin/bash

source ~/miniconda3/etc/profile.d/conda.sh
conda activate sf

# Create necessary directories if they do not already exist
mkdir -p data logs

# Define the datasets and levels
datasets=("cifar10" "cifar100" "stl10" "tinyimagenet")
levels=(4 5 6 7 8)

# Loop through datasets and levels to run the command
for dataset in "${datasets[@]}"; do
  for level in "${levels[@]}"; do
    python main.py --dataset "$dataset" --level "$level"
  done
done

conda deactivate