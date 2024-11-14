import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from models.mocogan import MoCoGAN
from src.config import Config

def evaluate_model(config):
    """Evaluate the trained MoCoGAN model."""
    # Initialize model
    model = MoCoGAN(config)
    model.setup_checkpointing()

    # Load the trained model weights
    model.checkpoint.restore(os.path.join(config.CHECKPOINT_DIR, 'mocogan_weights.h5'))

    # Generate samples
    content_noise = tf.random.normal([1, config.CONTENT_DIM])
    motion_noise = tf.random.normal([1, config.SEQUENCE_LENGTH, config.MOTION_DIM])

    generated_samples = model.generator(model.content_encoder(content_noise),
                                        model.motion_encoder(motion_noise))

    # Apply physics constraints (if any)
    generated_samples = model.physics.apply_constraints(generated_samples)

    # Reshape and plot the generated samples
    generated_samples = tf.reshape(generated_samples,
                                   [config.SEQUENCE_LENGTH, -1])
    plt.figure(figsize=(12, 4))
    for i in range(config.SEQUENCE_LENGTH):
        plt.subplot(1, config.SEQUENCE_LENGTH, i+1)
        plt.imshow(generated_samples[i])
        plt.axis('off')
    plt.show()

if __name__ == "__main__":
    config = Config()
    evaluate_model(config)