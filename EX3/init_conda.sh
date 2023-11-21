#!/usr/bin/env bash

# Script to init conda. Installs and switches to virtual environment

error() {
    echo -e "\e[0;31mERROR:\e[0m $1"
    exit 1
}

error "Script not functional yet..."

# Check if we're in the right directory
if [ ! -f "init_conda.sh" ]; then
    error "Run from EX3 directory"
fi

# Get myenvname from `environment.yml`
MY_ENV_NAME="$(grep "name:" environment.yml | sed 's/name: //g')"

# Check if `MY_ENV_NAME` == MLCMS_GroupI
if [ "$MY_ENV_NAME" != "MLCMS_GroupI" ]; then
    error "Environment name \"$MY_ENV_NAME\" has changed from MLCMS_GroupI. Error?"
fi

# Create conca env if doesn't exist
if [ ! $(conda env list | grep "$MY_ENV_NAME") ]; then
    conda env create -f environment.yml || error "Failed to create conda environment"
fi

# Activate conda environment
conda activate "$MY_ENV_NAME" || error "Failed to activate conda environment"

# Check if pip is from conda environment
if [ ! $(which pip | grep "$MY_ENV_NAME") ]; then
    error "Pip is not from our conda environment"
fi

# Check for unmet dependencies
# TODO: Check if it works...
if ! conda env export > /dev/null; then
   # If there are unmet dependencies, update the environment
   conda env update --file environment.yml --prune
   echo "Updated conda environment"
fi

echo "Ready to go!"