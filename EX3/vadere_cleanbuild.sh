#!/usr/bin/env bash

# Wrap old script

(cd ../EX2/ && ./clean_build.sh) || echo "ERROR: Execute this script from EX3 directory in the repo" && exit 1
