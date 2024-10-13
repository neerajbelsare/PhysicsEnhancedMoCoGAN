import tensorflow as tf
import os

# Paths to the video-level data directories
data_dir = 'data/yt8m/video'
train_dir = os.path.join(data_dir, 'train')
validate_dir = os.path.join(data_dir, 'validate')
test_dir = os.path.join(data_dir, 'test')


def load_dataset(directory):
    """Load TFRecord files from a directory."""
    files = tf.data.Dataset.list_files(os.path.join(directory, '*.tfrecord'))
    return tf.data.TFRecordDataset(files)


def parse_example(example_proto):
    """Parse a single video-level example."""
    feature_description = {
        'id': tf.io.FixedLenFeature([], tf.string),
        'labels': tf.io.VarLenFeature(tf.int64),
        'mean_rgb': tf.io.FixedLenFeature([1024], tf.float32),
        'mean_audio': tf.io.FixedLenFeature([128], tf.float32)
    }
    features = tf.io.parse_single_example(example_proto, feature_description)

    video_id = features['id']
    labels = tf.sparse.to_dense(features['labels'])
    rgb_features = features['mean_rgb']
    audio_features = features['mean_audio']

    # Combine RGB and audio features
    combined_features = tf.concat([rgb_features, audio_features], axis=0)

    return video_id, combined_features, labels


def preprocess_dataset(dataset):
    """Preprocess the dataset by parsing examples and batching."""
    return dataset.map(parse_example).batch(32)


# Load and preprocess datasets
train_dataset = load_dataset(train_dir)
validate_dataset = load_dataset(validate_dir)
test_dataset = load_dataset(test_dir)

train_dataset = preprocess_dataset(train_dataset)
validate_dataset = preprocess_dataset(validate_dataset)
test_dataset = preprocess_dataset(test_dataset)

# Example of iterating through the dataset
for video_ids, features, labels in train_dataset.take(1):
    print("Video IDs shape:", video_ids.shape)
    print("Features shape:", features.shape)
    print("Labels shape:", labels.shape)

# Calculate the number of examples in each dataset
train_size = sum(1 for _ in train_dataset)
validate_size = sum(1 for _ in validate_dataset)
test_size = sum(1 for _ in test_dataset)

print(f"Number of training examples: {train_size}")
print(f"Number of validation examples: {validate_size}")
print(f"Number of test examples: {test_size}")