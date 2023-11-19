#!/usr/bin/env bash

# Script to init venv. Installs

error() {
    echo "ERROR: $1"
    exit 1
}

# Check if we're in the right directory
if [ ! -f "init_venv.sh" ]; then
    error "Run from EX3 directory"
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    python3 -m venv .venv || error "Failed to create virtual environment"
fi

# Activate virtual environment
source .venv/bin/activate || error "Failed to activate virtual environment"

# Sanity check: Check if pip is from our virtual environment
if [ ! $(which pip | grep EX3/.venv) ]; then
    error "Pip is not from virtual environment"
fi

# Install requirements only if needed
python3 -c "import pkg_resources; pkg_resources.require(open('requirements.txt',mode='r'))" &>/dev/null \
  && echo "Requirements already installed" \
 || .venv/bin/pip install -r requirements.txt || error "Failed to install requirements"