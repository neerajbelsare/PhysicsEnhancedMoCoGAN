import sys
import os

from config import Config

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import tensorflow as tf
import numpy as np
import glob
from tqdm import tqdm

from models.mocogan import MoCoGAN

def load_data_chunk(chunk_path):
    """Load a single data chunk."""
    data = np.load(chunk_path)
    return data['features'], data['labels']

def train(config):
    """Main training function."""
    # Initialize model
    model = MoCoGAN(config)
    model.setup_checkpointing()  # This restores from the latest checkpoint, if available

    # Training loop
    step = model.checkpoint.save_counter.numpy()  # Retrieve last saved step count if available

    for epoch in range(config.EPOCHS):
        print(f"Epoch {epoch + 1}/{config.EPOCHS}")

        # Get all chunk files
        chunk_files = glob.glob(os.path.join(config.DATA_DIR, 'train_*.npz'))

        for chunk_file in tqdm(chunk_files, desc="Processing chunks"):
            # Load and preprocess chunk
            features, _ = load_data_chunk(chunk_file)

            # Create dataset from chunk
            dataset = tf.data.Dataset.from_tensor_slices(features)
            dataset = dataset.shuffle(10000).batch(config.BATCH_SIZE)

            for batch in dataset:
                # Reshape batch into sequences
                batch_sequences = tf.reshape(batch,
                                             [-1, config.SEQUENCE_LENGTH, batch.shape[-1]])

                # Training step
                gen_loss, disc_loss = model.train_step(batch_sequences)

                if step % 100 == 0:
                    print(f"Step {step}: Gen Loss: {gen_loss:.4f}, "
                          f"Disc Loss: {disc_loss:.4f}")

                # Save checkpoint
                if step % config.CHECKPOINT_INTERVAL == 0:
                    save_path = model.manager.save()
                    print(f"Saved checkpoint: {save_path}")

                step += 1

if __name__ == "__main__":
    config = Config()
    train(config)