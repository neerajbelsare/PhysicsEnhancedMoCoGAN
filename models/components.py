import tensorflow as tf
from tensorflow.keras import layers

class ContentEncoder(tf.keras.Model):
    def __init__(self, latent_dim):
        super(ContentEncoder, self).__init__()
        self.latent_dim = latent_dim

        self.encoder = tf.keras.Sequential([
            layers.Dense(512, activation='leaky_relu'),
            layers.Dense(256, activation='leaky_relu'),
            layers.Dense(latent_dim)
        ])

    def call(self, x):
        return self.encoder(x)

class MotionEncoder(tf.keras.Model):
    def __init__(self, motion_dim):
        super(MotionEncoder, self).__init__()
        self.motion_dim = motion_dim

        self.gru = layers.GRU(units=motion_dim, return_sequences=True)

    def call(self, x):
        return self.gru(x)

class Generator(tf.keras.Model):
    def __init__(self, content_dim, motion_dim):
        super(Generator, self).__init__()
        self.content_dim = content_dim
        self.motion_dim = motion_dim

        self.decoder = tf.keras.Sequential([
            layers.Dense(256, activation='leaky_relu'),
            layers.Dense(512, activation='leaky_relu'),
            layers.Dense(1152)  # 1024 (RGB) + 128 (audio)
        ])

    def call(self, content, motion):
        # Expand content to match sequence dimension
        batch_size = tf.shape(motion)[0]
        seq_length = tf.shape(motion)[1]

        # Repeat content for each timestep
        content_expanded = tf.expand_dims(content, axis=1)  # [batch, 1, content_dim]
        content_tiled = tf.tile(content_expanded, [1, seq_length, 1])  # [batch, seq_length, content_dim]

        # Concatenate along feature dimension
        combined = tf.concat([content_tiled, motion], axis=-1)  # [batch, seq_length, content_dim + motion_dim]

        # Process each timestep
        outputs = []
        for t in range(seq_length):
            timestep_features = combined[:, t, :]  # [batch, content_dim + motion_dim]
            output = self.decoder(timestep_features)  # [batch, 1152]
            outputs.append(output)

        # Stack timesteps
        return tf.stack(outputs, axis=1)  # [batch, seq_length, 1152]

class Discriminator(tf.keras.Model):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.frame_disc = tf.keras.Sequential([
            layers.Dense(512, activation='leaky_relu'),
            layers.Dense(256, activation='leaky_relu'),
            layers.Dense(1)
        ])

        self.video_disc = tf.keras.Sequential([
            layers.GRU(256, return_sequences=True),
            layers.GRU(128),
            layers.Dense(1)
        ])

    def call(self, x):
        # Process each frame
        frame_scores = []
        for t in range(tf.shape(x)[1]):
            frame = x[:, t, :]
            frame_score = self.frame_disc(frame)
            frame_scores.append(frame_score)
        frame_scores = tf.stack(frame_scores, axis=1)

        # Process whole sequence
        video_score = self.video_disc(x)

        return frame_scores, video_score