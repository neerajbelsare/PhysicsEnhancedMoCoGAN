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
if [ -z "$1" ]; then
  echo "Error: No stage specified."
  print_usage
  exit 1
fi

# Set the stage
STAGE=$1

# Create directories if they don't exist
mkdir -p data/processed logs checkpoints results/videos results/frames results/plots

# Main execution flow
case $STAGE in
  "preprocess")
    echo "Preprocessing data..."
    python scripts/preprocess_data.py \
      --train_dir data/yt8m/frame/train \
      --test_dir data/yt8m/frame/test \
      --validate_dir data/yt8m/frame/validate \
      --output_dir data/processed \
      --log logs/preprocess_log.txt
    ;;
  "train")
    echo "Training the MoCoGAN model..."
    python scripts/train_mocogan.py \
      --data_dir data/processed \
      --save_dir checkpoints \
      --log logs/training_log.txt \
      --epochs 50
    ;;
  "evaluate")
    echo "Evaluating the MoCoGAN model..."
    python scripts/evaluate_mocogan.py \
      --checkpoint checkpoints/epoch_10.pth \
      --data_dir data/processed \
      --results_dir results \
      --log logs/evaluation_log.txt
    ;;
  "generate")
    echo "Generating videos using the trained model..."
    python scripts/generate_video.py \
      --checkpoint checkpoints/epoch_10.pth \
      --results_dir results/videos \
      --num_videos 5 \
      --log logs/generation_log.txt
    ;;
  "physics_sim")
    echo "Running physics simulation..."
    python physics_engine/pybullet_simulation.py \
      --urdf_file physics_engine/pendulum.urdf \
      --log logs/physics_simulation_log.txt
    ;;
  *)
    echo "Error: Invalid stage specified."
    print_usage
    exit 1
    ;;
esac

echo "Execution of stage '$STAGE' completed."
