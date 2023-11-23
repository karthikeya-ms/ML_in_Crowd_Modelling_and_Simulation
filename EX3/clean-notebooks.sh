#!/usr/bin/env bash

# Script to clean ipynb notebooks under notebooks/

error() {
    echo -e "$(tput setaf 1)ERROR:$(tput sgr0) $1"
    exit 1
}

success(){
     echo -e "$(tput setaf 2)SUCCESS:$(tput sgr0) $1"
}

# Check if `nb-clean` is in path
if [ ! "$(which nb-clean)" ]; then
    error "nb-clean not in path. Install with 'pip install nb-clean'"
fi

# Check if we're in the right directory
if [ ! -f init_conda.sh ]; then
    error "Run from EX3 directory"
fi

# Clean every ipynb notebook in the notebooks directory
for notebook in notebooks/*.ipynb; do
    old_size=$(du -h "$notebook" | cut -f1)
    nb-clean clean "$notebook" || error "Failed to clean $notebook"
    new_size=$(du -h "$notebook" | cut -f1)
    success "Cleaned $notebook. Size difference: $old_size -> $new_size"
done

