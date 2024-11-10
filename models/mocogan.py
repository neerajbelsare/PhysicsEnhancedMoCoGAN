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

        # Add gradient clipping to optimizers
        self.gen_optimizer = tf.keras.optimizers.Adam(
            self.config.LEARNING_RATE,
            beta_1=self.config.BETA1,
            clipnorm=1.0
        )
        self.disc_optimizer = tf.keras.optimizers.Adam(
            self.config.LEARNING_RATE,
            beta_1=self.config.BETA1,
            clipnorm=1.0
        )

        # Build models
        self._build_models()

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

    def _compute_l2_loss(self, variables):
        """Compute L2 regularization loss for a list of variables."""
        l2_loss = 0.0
        for var in variables:
            if var is not None:
                l2_loss += tf.reduce_sum(tf.square(tf.cast(var, tf.float32)))
        return l2_loss

    def _build_models(self):
        """Build all models with dummy inputs."""
        batch_size = 2
        seq_length = self.config.SEQUENCE_LENGTH

        # Dummy inputs
        content_noise = tf.random.normal([batch_size, self.config.CONTENT_DIM], stddev=0.1)
        motion_noise = tf.random.normal(
            [batch_size, seq_length, self.config.MOTION_DIM],
            stddev=0.1
        )
        dummy_sequence = tf.random.normal([batch_size, seq_length, 1152])

        # Forward pass to build models
        fake_content = self.content_encoder(content_noise)
        fake_motion = self.motion_encoder(motion_noise)
        fake_sequences = self.generator(fake_content, fake_motion)
        real_frame_scores, real_video_score = self.discriminator(dummy_sequence)
        fake_frame_scores, fake_video_score = self.discriminator(fake_sequences)

        # Initialize optimizers with a dummy training step
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # Generator forward pass
            fake_content = self.content_encoder(content_noise)
            fake_motion = self.motion_encoder(motion_noise)
            fake_sequences = self.generator(fake_content, fake_motion)
            fake_sequences = self.physics.apply_constraints(fake_sequences)

            # Discriminator forward pass
            real_frame_scores, real_video_score = self.discriminator(dummy_sequence)
            fake_frame_scores, fake_video_score = self.discriminator(fake_sequences)

            # Calculate losses
            gen_loss = self._generator_loss(fake_frame_scores, fake_video_score)
            disc_loss = self._discriminator_loss(
                real_frame_scores, real_video_score,
                fake_frame_scores, fake_video_score
            )

        # Get variables
        gen_vars = (
                self.generator.trainable_variables +
                self.content_encoder.trainable_variables +
                self.motion_encoder.trainable_variables
        )
        disc_vars = self.discriminator.trainable_variables

        # Calculate and apply gradients
        gen_grads = gen_tape.gradient(gen_loss, gen_vars)
        disc_grads = disc_tape.gradient(disc_loss, disc_vars)

        self.gen_optimizer.apply_gradients(zip(gen_grads, gen_vars))
        self.disc_optimizer.apply_gradients(zip(disc_grads, disc_vars))

    def setup_checkpointing(self):
        """Setup checkpoint manager and restore if checkpoint exists."""
        self.manager = tf.train.CheckpointManager(
            self.checkpoint, self.config.CHECKPOINT_DIR, max_to_keep=3)

        if self.manager.latest_checkpoint:
            self.checkpoint.restore(self.manager.latest_checkpoint)
            print(f"Restored from checkpoint: {self.manager.latest_checkpoint}")
        else:
            print("Initializing from scratch")

    @tf.function
    def train_step(self, real_sequences):
        """Training step with improved stability measures."""
        batch_size = tf.shape(real_sequences)[0]

        # Add noise to real samples
        real_noise = tf.random.normal(tf.shape(real_sequences), mean=0.0, stddev=0.01)
        real_sequences_noisy = real_sequences + real_noise

        # Generate random noise with smaller magnitude
        content_noise = tf.random.normal([batch_size, self.config.CONTENT_DIM], stddev=0.1)
        motion_noise = tf.random.normal(
            [batch_size, self.config.SEQUENCE_LENGTH, self.config.MOTION_DIM],
            stddev=0.1
        )

        # Train discriminator
        with tf.GradientTape() as disc_tape:
            # Generate fake samples
            fake_content = self.content_encoder(content_noise)
            fake_motion = self.motion_encoder(motion_noise)
            fake_sequences = self.generator(fake_content, fake_motion)
            fake_sequences = self.physics.apply_constraints(fake_sequences)

            # Get discriminator outputs
            real_frame_scores, real_video_score = self.discriminator(real_sequences_noisy)
            fake_frame_scores, fake_video_score = self.discriminator(fake_sequences)

            # Calculate discriminator loss
            disc_loss = self._discriminator_loss(
                real_frame_scores, real_video_score,
                fake_frame_scores, fake_video_score
            )

        # Train generator
        with tf.GradientTape() as gen_tape:
            # Generate fake samples
            fake_content = self.content_encoder(content_noise)
            fake_motion = self.motion_encoder(motion_noise)
            fake_sequences = self.generator(fake_content, fake_motion)
            fake_sequences = self.physics.apply_constraints(fake_sequences)

            # Get discriminator outputs for fake samples
            fake_frame_scores, fake_video_score = self.discriminator(fake_sequences)

            # Calculate generator loss
            gen_loss = self._generator_loss(fake_frame_scores, fake_video_score)

        # Get variables
        gen_vars = (
                self.generator.trainable_variables +
                self.content_encoder.trainable_variables +
                self.motion_encoder.trainable_variables
        )
        disc_vars = self.discriminator.trainable_variables

        # Calculate gradients
        gen_grads = gen_tape.gradient(gen_loss, gen_vars)
        disc_grads = disc_tape.gradient(disc_loss, disc_vars)

        # Check for NaN gradients
        gen_grads_nan = [tf.reduce_any(tf.math.is_nan(g)) for g in gen_grads if g is not None]
        disc_grads_nan = [tf.reduce_any(tf.math.is_nan(g)) for g in disc_grads if g is not None]

        if not tf.reduce_any(gen_grads_nan):
            self.gen_optimizer.apply_gradients(zip(gen_grads, gen_vars))

        if not tf.reduce_any(disc_grads_nan):
            self.disc_optimizer.apply_gradients(zip(disc_grads, disc_vars))

        return gen_loss, disc_loss

    def _generator_loss(self, fake_frame_scores, fake_video_score):
        """Calculate generator loss with proper L2 regularization."""
        # Standard adversarial loss
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

        # Calculate L2 regularization for each network
        l2_coeff = 0.0001
        gen_l2 = self._compute_l2_loss(self.generator.trainable_variables)
        content_l2 = self._compute_l2_loss(self.content_encoder.trainable_variables)
        motion_l2 = self._compute_l2_loss(self.motion_encoder.trainable_variables)

        # Combine losses
        total_loss = frame_loss + video_loss + l2_coeff * (gen_l2 + content_l2 + motion_l2)

        return total_loss

    def _discriminator_loss(self, real_frame_scores, real_video_score,
                            fake_frame_scores, fake_video_score):
        """Calculate discriminator loss with label smoothing."""
        real_labels = tf.ones_like(real_frame_scores) * 0.9  # Label smoothing
        fake_labels = tf.zeros_like(fake_frame_scores) + 0.1  # Label smoothing

        real_frame_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=real_labels,
                logits=real_frame_scores
            )
        )
        real_video_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.ones_like(real_video_score) * 0.9,
                logits=real_video_score
            )
        )
        fake_frame_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=fake_labels,
                logits=fake_frame_scores
            )
        )
        fake_video_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                labels=tf.zeros_like(fake_video_score) + 0.1,
                logits=fake_video_score
            )
        )
        return real_frame_loss + real_video_loss + fake_frame_loss + fake_video_loss
