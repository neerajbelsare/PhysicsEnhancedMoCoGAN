import sys

import tensorflow as tf
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mocogan import MoCoGAN
from src.config import Config


def export_model_from_checkpoint(config, model_class, checkpoint_dir):
    """
    Load a model from checkpoint and save it as a SavedModel or HDF5.

    Parameters:
    - config: Configuration object with model parameters
    - model_class: Class of the model to be initialized
    - checkpoint_dir: Directory where checkpoint files are stored
    """
    # Initialize your model and restore from checkpoint
    model = model_class(config)
    checkpoint = tf.train.Checkpoint(
        content_encoder=model.content_encoder,
        motion_encoder=model.motion_encoder,
        generator=model.generator,
        discriminator=model.discriminator,
        gen_optimizer=model.gen_optimizer,
        disc_optimizer=model.disc_optimizer
    )
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)

    if latest_checkpoint:
        checkpoint.restore(latest_checkpoint).expect_partial()
        print(f"Checkpoint restored from {latest_checkpoint}")

        # Export the generator model
        export_path = os.path.join(checkpoint_dir, "final_model.h5")
        model.generator.save(export_path)  # SavedModel format
        print(f"Model exported to: {export_path}")
    else:
        print("No checkpoint found in the specified directory.")

if __name__ == "__main__":
    config = Config()
    export_model_from_checkpoint(config, MoCoGAN, "../checkpoints/20241111-153237")
