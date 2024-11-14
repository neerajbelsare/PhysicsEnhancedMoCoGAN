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

    # Paths
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    CHECKPOINT_DIR = os.path.join(
        BASE_DIR,
        'checkpoints',
        '20241111-153237'
        # datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    # Create required directories
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)