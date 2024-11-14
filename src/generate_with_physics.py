import sys
import tensorflow as tf
import numpy as np
import pybullet as p
import pybullet_data
import os
import cv2

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.mocogan import MoCoGAN
from config import Config

class PhysicsConstraints:
    def __init__(self):
        # Initialize PyBullet in DIRECT mode for headless operation
        self.client_id = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -9.8)  # Apply gravity

        # Load plane and sample rigid bodies
        self.plane_id = p.loadURDF("plane.urdf")
        self.rigid_body_id = p.loadURDF("r2d2.urdf", [0, 0, 0.5], useFixedBase=False)

    def apply_constraints(self, sequences):
        """Apply rigid body dynamics and collisions."""
        constrained_sequences = []

        for frame in sequences:
            # Reshape frame to fit PyBullet environment dimensions
            frame = np.array(frame).reshape(-1, 3)  # Adjust as necessary
            for i, pos in enumerate(frame):
                # Set positions in PyBullet (scaled to match model)
                p.resetBasePositionAndOrientation(self.rigid_body_id, pos, [0, 0, 0, 1])
                p.stepSimulation()

            # Capture final positions after constraints
            constrained_frame, _ = p.getBasePositionAndOrientation(self.rigid_body_id)
            constrained_sequences.append(constrained_frame)

        return np.array(constrained_sequences)

    def close(self):
        p.disconnect()

def apply_physics_to_model(config):
    # Initialize MoCoGAN and restore the checkpoint
    model = MoCoGAN(config)
    model.setup_checkpointing()

    # Initialize the physics engine
    physics_engine = PhysicsConstraints()

    # Load the latest checkpoint and apply physics constraints
    if model.manager.latest_checkpoint:
        model.checkpoint.restore(model.manager.latest_checkpoint)
        print(f"Restored model from checkpoint: {model.manager.latest_checkpoint}")
    else:
        raise ValueError("No checkpoint found. Please train the model first.")

    # Generate sequences and apply physics constraints
    content_noise = tf.random.normal([1, config.CONTENT_DIM], stddev=0.1)
    motion_noise = tf.random.normal([1, config.SEQUENCE_LENGTH, config.MOTION_DIM], stddev=0.1)

    fake_content = model.content_encoder(content_noise)
    fake_motion = model.motion_encoder(motion_noise)
    generated_sequences = model.generator(fake_content, fake_motion)

    # Convert TensorFlow tensor to numpy array for PyBullet compatibility
    generated_sequences = generated_sequences.numpy().squeeze(axis=0)

    # Apply physics constraints to the generated sequences
    constrained_sequences = physics_engine.apply_constraints(generated_sequences)

    # Save the final physics-enhanced video
    save_video(constrained_sequences, config.OUTPUT_VIDEO_PATH, config.FRAME_RATE)
    physics_engine.close()
    print("Final physics-enhanced video saved.")

def save_video(frames, output_path, frame_rate=30):
    """Save a sequence of frames as a video."""
    height, width = 256, 256  # Define based on your generated output
    video_writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))

    for frame in frames:
        # Ensure the frame is scaled to match the 0-255 RGB range
        frame_img = np.clip((frame * 255), 0, 255).astype(np.uint8)

        # Ensure the frame has three channels (RGB)
        if frame_img.shape[-1] == 1:  # If grayscale, convert to RGB
            frame_img = cv2.cvtColor(frame_img, cv2.COLOR_GRAY2RGB)
        elif frame_img.shape[-1] == 3:  # If already RGB
            frame_img = cv2.cvtColor(frame_img, cv2.COLOR_BGR2RGB)

        # Resize frame to the output dimensions if needed
        frame_img = cv2.resize(frame_img, (width, height))
        video_writer.write(frame_img)

    video_writer.release()
    print(f"Video saved to {output_path}")

if __name__ == "__main__":
    config = Config()
    apply_physics_to_model(config)
