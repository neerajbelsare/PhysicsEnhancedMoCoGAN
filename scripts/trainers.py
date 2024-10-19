import tensorflow as tf
import os


class MoCoGANTrainer:
    def __init__(self, generator, image_discriminator, video_discriminator, latent_dim, checkpoint_dir='./checkpoints'):
        self.generator = generator
        self.image_discriminator = image_discriminator
        self.video_discriminator = video_discriminator
        self.latent_dim = latent_dim
        self.checkpoint_dir = checkpoint_dir

        self.g_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
        self.d_optimizer = tf.keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)

        self.cross_entropy = tf.keras.losses.BinaryCrossentropy()

        # Checkpoint
        self.checkpoint = tf.train.Checkpoint(generator=self.generator,
                                              image_discriminator=self.image_discriminator,
                                              video_discriminator=self.video_discriminator,
                                              g_optimizer=self.g_optimizer,
                                              d_optimizer=self.d_optimizer)
        self.manager = tf.train.CheckpointManager(self.checkpoint, self.checkpoint_dir, max_to_keep=3)

    @tf.function
    def train_step(self, real_videos):
        batch_size = tf.shape(real_videos)[0]
        noise = tf.random.normal([batch_size, self.latent_dim])

        with tf.GradientTape(persistent=True) as tape:
            # Generate fake videos
            fake_videos = self.generator(noise, training=True)

            # Image discriminator
            real_images = tf.reshape(real_videos, [-1, *real_videos.shape[2:]])
            fake_images = tf.reshape(fake_videos, [-1, *fake_videos.shape[2:]])

            real_image_output = self.image_discriminator(real_images, training=True)
            fake_image_output = self.image_discriminator(fake_images, training=True)

            image_d_loss = self.cross_entropy(tf.ones_like(real_image_output), real_image_output) + \
                           self.cross_entropy(tf.zeros_like(fake_image_output), fake_image_output)

            # Video discriminator
            real_video_output = self.video_discriminator(real_videos, training=True)
            fake_video_output = self.video_discriminator(fake_videos, training=True)

            video_d_loss = self.cross_entropy(tf.ones_like(real_video_output), real_video_output) + \
                           self.cross_entropy(tf.zeros_like(fake_video_output), fake_video_output)

            # Generator loss
            g_loss = self.cross_entropy(tf.ones_like(fake_image_output), fake_image_output) + \
                     self.cross_entropy(tf.ones_like(fake_video_output), fake_video_output)

        # Compute gradients
        g_gradients = tape.gradient(g_loss, self.generator.trainable_variables)
        image_d_gradients = tape.gradient(image_d_loss, self.image_discriminator.trainable_variables)
        video_d_gradients = tape.gradient(video_d_loss, self.video_discriminator.trainable_variables)

        # Apply gradients
        self.g_optimizer.apply_gradients(zip(g_gradients, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(image_d_gradients, self.image_discriminator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(video_d_gradients, self.video_discriminator.trainable_variables))

        return g_loss, image_d_loss, video_d_loss

    def train(self, dataset, epochs, steps_per_epoch):
        # Restore the latest checkpoint if it exists
        if self.manager.latest_checkpoint:
            self.checkpoint.restore(self.manager.latest_checkpoint)
            print(f"Restored from {self.manager.latest_checkpoint}")
        else:
            print("Initializing from scratch.")

        for epoch in range(epochs):
            for step, batch in enumerate(dataset.take(steps_per_epoch)):
                g_loss, image_d_loss, video_d_loss = self.train_step(batch[0])  # batch[0] contains the video data

                if step % 100 == 0:
                    print(
                        f"Epoch {epoch + 1}, Step {step}, G Loss: {g_loss:.4f}, Image D Loss: {image_d_loss:.4f}, Video D Loss: {video_d_loss:.4f}")

            # Save checkpoint at the end of each epoch
            save_path = self.manager.save()
            print(f"Saved checkpoint for epoch {epoch + 1}: {save_path}")

            # Generate and save samples
            self.generate_and_save_samples(epoch + 1)

    def generate_and_save_samples(self, epoch):
        noise = tf.random.normal([1, self.latent_dim])
        generated_video = self.generator(noise, training=False)
        # Implement your logic to save the generated video
        print(f"Generated video sample saved for epoch {epoch}")

    def save_models(self, save_dir):
        self.generator.save(os.path.join(save_dir, 'generator.h5'))
        self.image_discriminator.save(os.path.join(save_dir, 'image_discriminator.h5'))
        self.video_discriminator.save(os.path.join(save_dir, 'video_discriminator.h5'))
        print(f"Models saved in {save_dir}")