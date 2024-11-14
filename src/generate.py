import sys
import os
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.pyplot as plt

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mocogan import MoCoGAN
from src.config import Config

def extract_frames_from_video(video_path, num_frames=16, frame_size=(32, 32)):
    """
    Extract frames from a video file.
    Args:
        video_path (str): Path to the video file.
        num_frames (int): Number of frames to extract.
        frame_size (tuple): Resize frames to this size.
    Returns:
        frames (np.ndarray): Array of resized frames.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = max(1, total_frames // num_frames)

    frame_idx = 0
    while count < num_frames and frame_idx < total_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            break

        # Convert to grayscale and resize
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frame = cv2.resize(frame, frame_size)

        # Extract RGB and audio features (simulated for this example)
        rgb_features = frame.flatten() / 255.0  # Normalize to [0,1]
        audio_features = np.zeros(128)  # Placeholder audio features

        # Combine features
        combined_features = np.concatenate([rgb_features, audio_features])
        frames.append(combined_features)

        count += 1
        frame_idx += frame_interval

    cap.release()
    return np.array(frames)

def process_video_for_prediction(video_path, config):
    """
    Process video frames for input to the model.
    """
    # Extract frames
    input_sequence = extract_frames_from_video(
        video_path,
        num_frames=config.SEQUENCE_LENGTH,
        frame_size=(32, 32)
    )

    # Reshape for model input
    input_sequence = tf.convert_to_tensor(input_sequence, dtype=tf.float32)
    input_sequence = tf.expand_dims(input_sequence, 0)  # Add batch dimension
    return input_sequence

def process_features_for_visualization(features):
    """
    Process the 1152-dimensional features into a visualizable format.
    Args:
        features: tensor of shape (1152,) containing RGB (1024) and audio (128) features
    Returns:
        rgb_image: reshaped RGB features as a 32x32x1 image
    """
    # Split RGB and audio features
    rgb_features = features[:1024]

    # Reshape RGB features into a square image
    rgb_image = tf.reshape(rgb_features, (32, 32, 1))

    # Normalize to [0, 1] range for visualization
    rgb_image = (rgb_image - tf.reduce_min(rgb_image)) / (tf.reduce_max(rgb_image) - tf.reduce_min(rgb_image))

    return rgb_image.numpy()

def predict_future_frames(model, input_sequence, num_future_frames=8):
    """
    Predict future frames given an input sequence.
    """
    # Extract content and motion features from input sequence
    content_features = model.content_encoder(input_sequence)
    motion_features = model.motion_encoder(input_sequence)

    # Generate future motion features
    future_motion = []
    current_motion = motion_features[:, -1, :]  # Get last motion state

    for _ in range(num_future_frames):
        # Predict next motion state using GRU
        current_motion = model.motion_encoder.gru(
            tf.expand_dims(current_motion, 1)
        )[:, -1, :]
        future_motion.append(current_motion)

    future_motion = tf.stack(future_motion, axis=1)

    # Generate future frames
    future_frames = model.generator(
        content_features,
        tf.concat([motion_features, future_motion], axis=1)
    )

    return future_frames

def visualize_prediction(input_sequence, predicted_sequence, save_path=None):
    """
    Visualize input and predicted frames.
    """
    total_frames = input_sequence.shape[1] + predicted_sequence.shape[1]

    plt.figure(figsize=(20, 4))

    # Plot input sequence
    for i in range(input_sequence.shape[1]):
        plt.subplot(2, total_frames, i + 1)
        frame = process_features_for_visualization(input_sequence[0, i])
        plt.imshow(frame, cmap='gray')
        plt.title(f'Input {i+1}')
        plt.axis('off')

    # Plot predicted sequence
    for i in range(predicted_sequence.shape[1]):
        plt.subplot(2, total_frames, input_sequence.shape[1] + i + 1)
        frame = process_features_for_visualization(predicted_sequence[0, i])
        plt.imshow(frame, cmap='gray')
        plt.title(f'Predicted {i+1}')
        plt.axis('off')

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)
        print(f"Visualization saved to: {save_path}")

    plt.show()

def main():
    # Initialize configuration and model
    config = Config()
    model = MoCoGAN(config)
    model.setup_checkpointing()

    # Load the latest checkpoint
    if model.manager.latest_checkpoint:
        model.checkpoint.restore(model.manager.latest_checkpoint)
        print(f"Restored from checkpoint: {model.manager.latest_checkpoint}")
    else:
        print("No checkpoint found. Unable to generate predictions.")
        return

    # Process input video
    video_path = "path/to/your/input/video.mp4"  # Replace with your video path
    input_sequence = process_video_for_prediction(video_path, config)

    # Predict future frames
    predicted_sequence = predict_future_frames(
        model,
        input_sequence,
        num_future_frames=8
    )

    # Visualize results
    visualize_prediction(
        input_sequence,
        predicted_sequence,
        save_path='predictions/predicted_frames.png'
    )

    # Save raw predictions if needed
    np.save('predictions/predicted_frames.npy', predicted_sequence.numpy())

if __name__ == "__main__":
    main()