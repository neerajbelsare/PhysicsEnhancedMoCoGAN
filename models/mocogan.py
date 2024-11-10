import tensorflow as tf
import numpy as np
import glob
import os

from models.components import ContentEncoder, MotionEncoder, Generator, Discriminator
from models.physics import PhysicsConstraints

class MoCoGAN:
    def __init__(self, config):
        self.config = config

        # Initialize networks
        self.content_encoder = ContentEncoder(config.CONTENT_DIM)
        self.motion_encoder = MotionEncoder(config.MOTION_DIM)
        self.generator = Generator(config.CONTENT_DIM, config.MOTION_DIM)
        self.discriminator = Discriminator()
        self.physics = PhysicsConstraints()

        # Initialize optimizers
        self.gen_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE, beta_1=config.BETA1)
        self.disc_optimizer = tf.keras.optimizers.Adam(config.LEARNING_RATE, beta_1=config.BETA1)

        # Initialize checkpoint
        self.checkpoint = tf.train.Checkpoint(
            content_encoder=self.content_encoder,
            motion_encoder=self.motion_encoder,
            generator=self.generator,
            discriminator=self.discriminator,
            gen_optimizer=self.gen_optimizer,
            disc_optimizer=self.disc_optimizer
        )
        self.manager = None

    def setup_checkpointing(self):
        """Setup checkpoint manager."""
        self.manager = tf.train.CheckpointManager(
            self.checkpoint, self.config.CHECKPOINT_DIR, max_to_keep=3)

        if self.manager.latest_checkpoint:
            self.checkpoint.restore(self.manager.latest_checkpoint)
            print(f"Restored from checkpoint: {self.manager.latest_checkpoint}")
        else:
            print("Initializing from scratch")

    @tf.function
    def train_step(self, real_sequences):
        """Single training step."""
        batch_size = tf.shape(real_sequences)[0]

        # Generate random noise
        content_noise = tf.random.normal([batch_size, self.config.CONTENT_DIM])
        motion_noise = tf.random.normal(
            [batch_size, self.config.SEQUENCE_LENGTH, self.config.MOTION_DIM]
        )

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generate fake sequences
            fake_content = self.content_encoder(content_noise)  # [batch, content_dim]
            fake_motion = self.motion_encoder(motion_noise)     # [batch, seq_length, motion_dim]
            fake_sequences = self.generator(fake_content, fake_motion)  # [batch, seq_length, 1152]

            # Apply physics constraints
            fake_sequences = self.physics.apply_constraints(fake_sequences)

            # Get discriminator outputs
            real_frame_scores, real_video_score = self.discriminator(real_sequences)
            fake_frame_scores, fake_video_score = self.discriminator(fake_sequences)

            # Calculate losses
            gen_loss = self._generator_loss(fake_frame_scores, fake_video_score)
            disc_loss = self._discriminator_loss(
                real_frame_scores, real_video_score,
                fake_frame_scores, fake_video_score
            )

        # Update weights
        self._apply_gradients(gen_tape, disc_tape, gen_loss, disc_loss)

        return gen_loss, disc_loss

    def _generator_loss(self, fake_frame_scores, fake_video_score):
        """Calculate generator loss."""
        frame_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(fake_frame_scores),
                logits=fake_frame_scores
            )
        )
        video_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(fake_video_score),
                logits=fake_video_score
            )
        )
        return frame_loss + video_loss

    def _discriminator_loss(self, real_frame_scores, real_video_score,
                            fake_frame_scores, fake_video_score):
        """Calculate discriminator loss."""
        real_frame_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(real_frame_scores),
                logits=real_frame_scores
            )
        )
        real_video_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(real_video_score),
                logits=real_video_score
            )
        )
        fake_frame_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(fake_frame_scores),
                logits=fake_frame_scores
            )
        )
        fake_video_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(fake_video_score),
                logits=fake_video_score
            )
        )
        return real_frame_loss + real_video_loss + fake_frame_loss + fake_video_loss

    def _apply_gradients(self, gen_tape, disc_tape, gen_loss, disc_loss):
        """Apply gradients to update model weights."""
        gen_gradients = gen_tape.gradient(gen_loss, [
            self.generator.trainable_variables,
            self.content_encoder.trainable_variables,
            self.motion_encoder.trainable_variables
        ])
        disc_gradients = disc_tape.gradient(disc_loss,
                                            self.discriminator.trainable_variables)

        self.gen_optimizer.apply_gradients(zip(gen_gradients[0],
                                               self.generator.trainable_variables))
        self.gen_optimizer.apply_gradients(zip(gen_gradients[1],
                                               self.content_encoder.trainable_variables))
        self.gen_optimizer.apply_gradients(zip(gen_gradients[2],
                                               self.motion_encoder.trainable_variables))
        self.disc_optimizer.apply_gradients(zip(disc_gradients,
                                                self.discriminator.trainable_variables))