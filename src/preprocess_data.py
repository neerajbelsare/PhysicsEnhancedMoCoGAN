import tensorflow as tf
import numpy as np
import os
from tqdm import tqdm
import multiprocessing
import glob

# Enable GPU memory growth to avoid allocating all GPU memory at once
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# Paths to the video-level data directories
data_dir = '../data/yt8m/video'
train_dir = os.path.join(data_dir, 'train')
validate_dir = os.path.join(data_dir, 'validate')
test_dir = os.path.join(data_dir, 'test')

# Output directories for preprocessed data
output_dir = '../data/yt8m/preprocessed'
os.makedirs(output_dir, exist_ok=True)


def load_dataset(directory):
    """Load TFRecord files from a directory."""
    files = tf.data.Dataset.list_files(os.path.join(directory, '*.tfrecord'))
    dataset = files.interleave(
        tf.data.TFRecordDataset,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=False
    )
    return dataset


@tf.function
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


@tf.function
def normalize_features(features):
    """Normalize features to have zero mean and unit variance."""
    mean, variance = tf.nn.moments(features, axes=[0])
    return (features - mean) / tf.sqrt(variance + 1e-8)


@tf.function
def pad_labels(video_id, features, labels):
    """Pad labels to a fixed size."""
    max_labels = 100  # Choose a reasonable maximum number of labels
    labels = tf.pad(labels, [[0, max_labels - tf.shape(labels)[0]]])
    labels = labels[:max_labels]
    return video_id, features, labels


def preprocess_dataset(dataset):
    """Preprocess the dataset by parsing examples, normalizing features, and padding labels."""
    return (dataset
            .map(parse_example, num_parallel_calls=tf.data.AUTOTUNE)
            .map(lambda id, feat, lab: (id, normalize_features(feat), lab), num_parallel_calls=tf.data.AUTOTUNE)
            .map(pad_labels, num_parallel_calls=tf.data.AUTOTUNE)
            .prefetch(tf.data.AUTOTUNE))


def process_chunk(chunk_data):
    """Process a chunk of data."""
    chunk_ids, chunk_features, chunk_labels = chunk_data
    return np.array(chunk_ids), np.array(chunk_features), np.array(chunk_labels)


def get_latest_chunk_number(output_dir, file_prefix):
    """Get the number of the latest chunk file."""
    existing_chunks = glob.glob(os.path.join(output_dir, f"{file_prefix}_*.npz"))
    if not existing_chunks:
        return -1
    latest_chunk = max(existing_chunks, key=os.path.getctime)
    return int(latest_chunk.split('_')[-1].split('.')[0])


def process_and_save_dataset(input_dir, output_file, batch_size=5000, chunk_size=20):
    """Process the dataset and save it to .npz files, resuming from the latest chunk if it exists."""
    dataset = load_dataset(input_dir)
    dataset = preprocess_dataset(dataset).batch(batch_size)

    # Get the latest chunk number
    latest_chunk = get_latest_chunk_number(output_dir, os.path.basename(output_file))
    start_chunk = latest_chunk + 1

    # Skip already processed batches
    dataset = dataset.skip(start_chunk * chunk_size)

    # Get the total number of remaining elements in the dataset
    total_elements = tf.data.experimental.cardinality(dataset).numpy()

    # Initialize lists to store chunks of processed data
    chunks = []

    # Process the remaining dataset in chunks
    for i, batch in enumerate(tqdm(dataset, total=total_elements, unit='batch', initial=start_chunk * chunk_size)):
        batch_ids, batch_features, batch_labels = batch
        chunks.append((batch_ids.numpy(), batch_features.numpy(), batch_labels.numpy()))

        # Process and save when we have accumulated 'chunk_size' batches
        if len(chunks) == chunk_size or i == total_elements - 1:
            with multiprocessing.Pool() as pool:
                processed_chunks = pool.map(process_chunk, chunks)

            # Concatenate processed chunks
            video_ids = np.concatenate([chunk[0] for chunk in processed_chunks])
            features = np.concatenate([chunk[1] for chunk in processed_chunks])
            labels = np.concatenate([chunk[2] for chunk in processed_chunks])

            # Save the processed data
            chunk_number = start_chunk + (i // chunk_size)
            np.savez_compressed(f"{output_file}_{chunk_number}.npz", video_ids=video_ids, features=features,
                                labels=labels)
            print(f"Saved preprocessed data chunk to {output_file}_{chunk_number}.npz")
            print(f"Chunk shape: {features.shape}")

            # Clear the chunks list
            chunks.clear()

    print(f"Finished processing and saving data to {output_file}_*.npz files")


# Process and save each dataset
process_and_save_dataset(train_dir, os.path.join(output_dir, 'train'))
process_and_save_dataset(validate_dir, os.path.join(output_dir, 'validate'))
process_and_save_dataset(test_dir, os.path.join(output_dir, 'test'))
