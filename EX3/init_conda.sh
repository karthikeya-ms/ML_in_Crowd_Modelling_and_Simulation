#!/usr/bin/env bash

# Script to init venv. Installs

error() {
    echo "ERROR: $1"
    exit 1
}

# Check if we're in the right directory
if [ ! -f "init_conda.sh" ]; then
    error "Run from EX3 directory"
fi

# Get myenvname from `environment.yml`
MY_ENV_NAME=$(grep "name:" environment.yml | sed 's/name: //g')

# Create conca venv if doesn't exist
if [ ! $(conda env list | grep "$MY_ENV_NAME") ]; then
    conda env create -f environment.yml || error "Failed to create conda environment"
fi

# Activate conda environment
conda activate "$MY_ENV_NAME" || error "Failed to activate conda environment"

# Check if pip is from conda environment
if [ ! $(which pip | grep "$MY_ENV_NAME") ]; then
    error "Pip is not from our conda environment"
fi

echo "Ready to go!"