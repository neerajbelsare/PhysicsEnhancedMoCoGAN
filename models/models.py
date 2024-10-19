import tensorflow as tf
from tensorflow.keras import layers, models

class Generator(tf.keras.Model):
    def __init__(self, latent_dim, num_frames, image_size, channels):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.num_frames = num_frames
        self.image_size = image_size
        self.channels = channels

        self.content_generator = self.build_content_generator()
        self.motion_generator = layers.LSTM(latent_dim, return_sequences=True)

    def build_content_generator(self):
        model = tf.keras.Sequential([
            layers.Dense(4 * 4 * 512, input_shape=(self.latent_dim,)),
            layers.Reshape((4, 4, 512)),
            layers.Conv2DTranspose(256, 4, strides=2, padding='same'),
            layers.LeakyReLU(),
            layers.Conv2DTranspose(128, 4, strides=2, padding='same'),
            layers.LeakyReLU(),
            layers.Conv2DTranspose(64, 4, strides=2, padding='same'),
            layers.LeakyReLU(),
            layers.Conv2DTranspose(self.channels, 4, strides=2, padding='same', activation='tanh')
        ])
        return model

    def call(self, inputs):
        batch_size = tf.shape(inputs)[0]
        content = self.content_generator(inputs)
        motion = self.motion_generator(tf.repeat(inputs[:, tf.newaxis, :], self.num_frames, axis=1))
        video = tf.map_fn(lambda x: self.content_generator(x), motion)
        return video

class ImageDiscriminator(tf.keras.Model):
    def __init__(self, image_size, channels):
        super(ImageDiscriminator, self).__init__()
        self.model = tf.keras.Sequential([
            layers.Conv2D(64, 4, strides=2, padding='same', input_shape=(image_size, image_size, channels)),
            layers.LeakyReLU(),
            layers.Conv2D(128, 4, strides=2, padding='same'),
            layers.LeakyReLU(),
            layers.Conv2D(256, 4, strides=2, padding='same'),
            layers.LeakyReLU(),
            layers.Flatten(),
            layers.Dense(1, activation='sigmoid')
        ])

    def call(self, inputs):
        return self.model(inputs)

class VideoDiscriminator(tf.keras.Model):
    def __init__(self, num_frames, image_size, channels):
        super(VideoDiscriminator, self).__init__()
        self.model = tf.keras.Sequential([
            layers.TimeDistributed(layers.Conv2D(64, 4, strides=2, padding='same'),
                                   input_shape=(num_frames, image_size, image_size, channels)),
            layers.TimeDistributed(layers.LeakyReLU()),
            layers.TimeDistributed(layers.Conv2D(128, 4, strides=2, padding='same')),
            layers.TimeDistributed(layers.LeakyReLU()),
            layers.TimeDistributed(layers.Conv2D(256, 4, strides=2, padding='same')),
            layers.TimeDistributed(layers.LeakyReLU()),
            layers.TimeDistributed(layers.Flatten()),
            layers.LSTM(256),
            layers.Dense(1, activation='sigmoid')
        ])

    def call(self, inputs):
        return self.model(inputs)