# # utils/data_loader.py
#
# import os
# import numpy as np
# import pandas as pd
#
# def load_shard(shard_path):
#     """
#     Load a single shard file (e.g., .csv or .tfrecord) and return its data.
#
#     Args:
#         shard_path (str): Path to the shard file.
#
#     Returns:
#         data (pd.DataFrame or np.array): Loaded shard data.
#     """
#     if not os.path.exists(shard_path):
#         raise FileNotFoundError(f"Shard file not found: {shard_path}")
#
#     # Example: Loading a CSV file. Replace with TFRecord loading if needed.
#     if shard_path.endswith(".csv"):
#         data = pd.read_csv(shard_path)
#     elif shard_path.endswith(".tfrecord"):
#         # Placeholder for loading .tfrecord files (requires tensorflow or tfrecord library)
#         import tensorflow as tf
#         raw_dataset = tf.data.TFRecordDataset(shard_path)
#         data = [record for record in raw_dataset]
#     else:
#         raise ValueError(f"Unsupported file type: {shard_path}")
#
#     return data
#
# def batch_data(data, batch_size):
#     """
#     Split data into batches.
#
#     Args:
#         data (np.array or pd.DataFrame): The data to split into batches.
#         batch_size (int): The size of each batch.
#
#     Returns:
#         batches (list): List of batches.
#     """
#     num_batches = len(data) // batch_size
#     batches = [data[i * batch_size:(i + 1) * batch_size] for i in range(num_batches)]
#     return batches
#
# def normalize_data(data):
#     """
#     Normalize the data to have zero mean and unit variance.
#
#     Args:
#         data (pd.DataFrame or np.array): The data to normalize.
#
#     Returns:
#         normalized_data (pd.DataFrame or np.array): The normalized data.
#     """
#     normalized_data = (data - np.mean(data)) / np.std(data)
#     return normalized_data
