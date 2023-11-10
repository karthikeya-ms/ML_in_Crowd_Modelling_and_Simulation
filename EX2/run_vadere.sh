#!/usr/bin/env bash

# Convenience script for running. Uses default `java` in path
# Execute this script from the EX2 directory

on_error(){
    echo "ERROR: $1"
    exit 1
}

# Check if java is installed
command -v java >/dev/null 2>&1 || on_error "Java is not installed"

java -jar vadere/VadereGui/target/vadere-gui.jar || on_error "Build and execute from EX2 directory"

