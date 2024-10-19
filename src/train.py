import os
import sys
import tensorflow as tf

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from models.models import Generator, ImageDiscriminator, VideoDiscriminator
from scripts.data import YouTube8MDataset
from scripts.trainers import MoCoGANTrainer

# Hyperparameters
latent_dim = 100
num_frames = 16
image_size = 64
channels = 3
batch_size = 32
epochs = 100

# Data directory
data_dir = 'data/yt8m/preprocessed'

# Checkpoint and model save directories
checkpoint_dir = './checkpoints'
model_save_dir = './saved_models'

# Create dataset
dataset = YouTube8MDataset(data_dir, batch_size, num_frames=16, image_size=64, channels=3)
train_dataset = dataset.get_train_dataset()

# Calculate steps per epoch
steps_per_epoch = tf.data.experimental.cardinality(train_dataset).numpy()

# Create models
generator = Generator(latent_dim, num_frames, image_size, channels)
image_discriminator = ImageDiscriminator(image_size, channels)
video_discriminator = VideoDiscriminator(num_frames, image_size, channels)

# Create trainer with checkpoint directory
trainer = MoCoGANTrainer(generator, image_discriminator, video_discriminator, latent_dim, checkpoint_dir)

# Train the model
trainer.train(train_dataset, epochs, steps_per_epoch)

# Save the final models
trainer.save_models(model_save_dir)

print("Training completed and models saved.")