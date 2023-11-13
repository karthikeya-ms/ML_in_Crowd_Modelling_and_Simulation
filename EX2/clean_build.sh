#!/usr/bin/env bash

# Convenience script for building
# Execute this script from the EX2 directory

on_error(){
    echo "ERROR: $1"
    exit 1
}

# Check if maven is installed
command -v mvn >/dev/null 2>&1 || on_error "Maven is not installed"

(cd vadere || on_error "Please execute from EX2 directory"
mvn clean && mvn -D maven.test.skip=true package)

