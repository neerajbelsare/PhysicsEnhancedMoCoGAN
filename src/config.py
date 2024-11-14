import os
from datetime import datetime

class Config:
    # Model parameters
    CONTENT_DIM = 128
    MOTION_DIM = 64
    SEQUENCE_LENGTH = 16

    # Training parameters
    BATCH_SIZE = 32
    EPOCHS = 10
    CHECKPOINT_INTERVAL = 1000
    LEARNING_RATE = 2e-4
    BETA1 = 0.5

    # Data parameters
    DATA_DIR = 'data/yt8m/preprocessed'

    # Physics constraints
    APPLY_PHYSICS = True  # Set to True to apply physics constraints in final generation
    GRAVITY = -9.8  # Gravity for PyBullet physics engine (can be adjusted)

    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CHECKPOINT_DIR = os.path.join(
        BASE_DIR,
        'checkpoints',
        '20241111-153237'  # Ensure this directory exists or update with the correct checkpoint date
    )

    OUTPUT_DIR = os.path.join(BASE_DIR, 'output')
    OUTPUT_VIDEO_PATH = os.path.join(OUTPUT_DIR, 'final_generated_video.mp4')

    # Video generation parameters
    FRAME_RATE = 30  # Frame rate of the generated video

    # Create required directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
