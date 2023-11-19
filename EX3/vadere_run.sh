#!/usr/bin/env bash

# Wrap old script

(cd ../EX2/ && ./run_vadere.sh) || echo "ERROR during Vadere run. Exiting..." && exit 1
