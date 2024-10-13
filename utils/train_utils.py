# utils/train_utils.py

import os
import torch


def save_processed_data(processed_data, file_path):
    """
    Save the processed data to a file.

    Args:
        processed_data (pd.DataFrame or np.array): The data to save.
        file_path (str): The path to the file where data will be saved.

    Returns:
        None
    """
    if isinstance(processed_data, pd.DataFrame):
        processed_data.to_csv(file_path, index=False)
    elif isinstance(processed_data, np.ndarray):
        np.save(file_path, processed_data)
    else:
        raise ValueError("Unsupported data type for saving.")


def save_model_checkpoint(model, optimizer, epoch, save_dir):
    """
    Save a checkpoint of the model's current state during training.

    Args:
        model (torch.nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer state to save.
        epoch (int): The current epoch number.
        save_dir (str): Directory to save the checkpoint.

    Returns:
        None
    """
    os.makedirs(save_dir, exist_ok=True)
    checkpoint_path = os.path.join(save_dir, f"epoch_{epoch}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, checkpoint_path)
    print(f"Checkpoint saved at {checkpoint_path}")


def load_model_checkpoint(checkpoint_path, model, optimizer=None):
    """
    Load a model checkpoint from the specified file.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (torch.nn.Module): The model to load the state dictionary into.
        optimizer (torch.optim.Optimizer, optional): The optimizer to load the state dictionary into. Default is None.

    Returns:
        epoch (int): The epoch number at which the checkpoint was saved.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    epoch = checkpoint['epoch']
    print(f"Loaded checkpoint from {checkpoint_path} (Epoch {epoch})")
    return epoch
