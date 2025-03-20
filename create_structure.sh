#!/usr/bin/env bash

# Create directories
mkdir -p src slurm output/results output/logs output/figures tests

# Create Python files in src
touch src/MastodonData.py \
      src/main.py \
      src/util.py \
      src/analysis.py

# Create shell scripts in slurm
touch slurm/run_1node_1core.sh \
      slurm/run_1node_8cores.sh \
      slurm/run_2nodes_8cores.sh

# Create test files
touch tests/test_data_processing.py \
      tests/test_sentiment_analysis.py

# Create additional files
touch README.md requirements.txt

echo "Project structure created successfully."