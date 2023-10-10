#!/bin/bash

# Remove the old log file if it exists
rm -f test_all.txt

# Find all Python test files in the current directory
for file in test_*.py
do
    # Print the name of the test being run
    echo "Running $file" | tee -a test_all.txt

    # Run the test and append the output to the log file
    python $file 2>&1 | tee -a test_all.txt

    # Print a separator line
    echo "--------------------------------------------------" | tee -a test_all.txt
done