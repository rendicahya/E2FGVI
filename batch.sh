#!/bin/bash

if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <number_of_iterations>"
    exit 1
fi

NUM_ITERATIONS=$1

for ((i=1; i<=$NUM_ITERATIONS; i++)); do
    echo "Running batch-detector.py - iteration $i"
    python batch-detector.py
done