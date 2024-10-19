import os

import numpy as np
import tensorflow as tf


class YouTube8MDataset:
    def __init__(self, data_dir, batch_size, num_frames=16, image_size=64, channels=3):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_frames = num_frames
        self.image_size = image_size
        self.channels = channels
        self.expected_feature_size = num_frames * image_size * image_size * channels

    def npz_generator(self):
        """Generator that yields data from .npz files."""
        file_list = [os.path.join(self.data_dir, fname) for fname in os.listdir(self.data_dir) if
                     fname.endswith('.npz')]

        for file_path in file_list:
            with np.load(file_path) as data:
                features = data['features']  # Assuming this is the preprocessed video data
                labels = data['labels']  # Assuming this is the label data

                print(f"Loaded {file_path} with features shape: {features.shape}")  # Debugging

                # Check the shape of features
                if features.ndim == 2 and features.shape[1] == self.expected_feature_size:
                    features = np.reshape(features,
                                          (-1, self.num_frames, self.image_size, self.image_size, self.channels))
                    print(f"Reshaped features to: {features.shape}")  # Debugging
                else:
                    print("Warning: Unexpected features shape; expected (N, 196608), got: ", features.shape)

                yield features, labels

    def get_dataset(self):
        """Creates a TensorFlow Dataset from the .npz files generator."""
        dataset = tf.data.Dataset.from_generator(
            self.npz_generator,
            output_signature=(
                tf.TensorSpec(shape=(None, self.num_frames, self.image_size, self.image_size, self.channels),
                              dtype=tf.float32),
                tf.TensorSpec(shape=(None,), dtype=tf.int64)  # Adjust shape/dtype for labels as necessary
            )
        )
        # Shuffle, batch, and prefetch the dataset to avoid bottlenecks
        dataset = dataset.shuffle(buffer_size=1000)  # Adjust buffer size depending on data size
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

        return dataset

    def get_train_dataset(self):
        """Returns the training dataset."""
        return self.get_dataset()

    def get_test_dataset(self):
        """Returns the test dataset if needed."""
        return self.get_dataset()

    def get_validate_dataset(self):
        """Returns the validation dataset if needed."""
        return self.get_dataset()
