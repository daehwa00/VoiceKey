import random
import torch
import numpy as np

import os
import torch
from datetime import datetime


def set_seed(seed: int):
    """
    Set the random seed for PyTorch, NumPy, and Python's random module to ensure reproducibility.

    Args:
    - seed (int): The seed value to be set.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def save_model(model, optimizer, epoch, save_dir, model_name, current_time):
    """
    Save the model to the path directory provided
    """
    # Create a new directory path with the current time
    save_dir = os.path.join(save_dir, current_time)

    save_path = os.path.join(save_dir, model_name + "_epoch_" + str(epoch) + ".pt")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        save_path,
    )

    print(f"Model saved to ==> {save_path}")
