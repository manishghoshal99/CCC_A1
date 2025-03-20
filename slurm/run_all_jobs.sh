#!/bin/bash
# Script to run all Mastodon analysis jobs

# Default to largest file if no argument provided
DATA_FILE=${1:-"mastodon-144Gb.ndjson"}

# Create output directory
mkdir -p ./output
mkdir -p ./output/logs
mkdir -p ./output/results

echo "Running Mastodon analysis on $DATA_FILE"
echo "Submitting job for 1 node, 1 core..."
sbatch ./slurm/run_1node_1core.sh $DATA_FILE

# Wait a bit before submitting the next job to avoid overwhelming the scheduler
sleep 5

echo "Submitting job for 1 node, 8 cores..."
sbatch ./slurm/run_1node_8cores.sh $DATA_FILE

sleep 5

echo "Submitting job for 2 nodes, 8 cores (4 per node)..."
sbatch ./slurm/run_2nodes_8cores.sh $DATA_FILE

echo "All jobs submitted. Check status with 'squeue -u $USER'"
echo "Results will be saved in ./output/results/"