#!/bin/bash

# This script automates the different stages of the MoCoGAN project with physics constraints.
# Each stage corresponds to a specific script located in the `scripts/` directory.

# Function to print usage
print_usage() {
  echo "Usage: ./run.sh [stage]"
  echo "Stages:"
  echo "  preprocess   - Run data preprocessing using 'scripts/preprocess_data.py'"
  echo "  train        - Train the MoCoGAN model using 'scripts/train_mocogan.py'"
  echo "  evaluate     - Evaluate the trained model using 'scripts/evaluate_mocogan.py'"
  echo "  generate     - Generate videos using 'scripts/generate_video.py'"
  echo "  physics_sim  - Run physics simulation using 'physics_engine/pybullet_simulation.py'"
  echo ""
  echo "Example: ./run.sh preprocess"
}

# Check if a stage argument was provided
if [ $# -eq 0 ]; then
  echo "Error: No stage specified."
  print_usage
  exit 1
fi

# Set the stage
STAGE=$1

# Create directories if they don't exist
mkdir -p data/yt8m/preprocessed logs checkpoints results/videos results/frames results/plots

# Function to run a command and check its exit status
run_command() {
  echo "Running: $1"
  eval $1
  if [ $? -ne 0 ]; then
    echo "Error: Command failed. Check the logs for details."
    exit 1
  fi
}

# Main execution flow
case $STAGE in
  "preprocess")
    echo "Preprocessing data..."
    run_command "python src/preprocess_data.py";;
  "train")
    echo "Training the MoCoGAN model..."
    run_command "python src/train.py";;
  "evaluate")
    echo "Evaluating the MoCoGAN model..."
    run_command "python src/evaluate.py";;
  "generate")
    echo "Generating videos using the trained model..."
    run_command "python src/generate.py"
    ;;
  "physics_sim")
    echo "Running physics simulation..."
    run_command "python physics_engine/pybullet_simulation.py \
      --urdf_file physics_engine/pendulum.urdf \
      --log logs/physics_simulation_log.txt"
    ;;
  *)
    echo "Error: Invalid stage specified."
    print_usage
    exit 1
    ;;
esac

echo "Execution of stage '$STAGE' completed successfully."