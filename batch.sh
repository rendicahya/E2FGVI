#!/bin/bash

# Check if the number of iterations is provided as an argument
if [ "$#" -ne 1 ]; then
    echo "Usage: $0 <number_of_iterations>"
    exit 1
fi

# Get the number of iterations from the command-line argument
NUM_ITERATIONS=$1

# Run batch.py for the specified number of iterations
for ((i=1; i<=$NUM_ITERATIONS; i++)); do
    echo "Running batch.py - Iteration $i"
    python batch.py
done