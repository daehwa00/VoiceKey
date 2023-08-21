import random
import torch
import numpy as np

import os
import torch
from tqdm import tqdm
import wandb


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


def load_model(model, optimizer, save_dir, model_name):
    """
    Load the model from the path directory provided
    """
    load_path = os.path.join(save_dir, model_name + ".pt")
    if os.path.exists(load_path):
        checkpoint = torch.load(load_path)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        print(f"Model loaded from <== {load_path}")

        return model, optimizer, checkpoint["epoch"]
    else:
        print(f"WARNING: No model found in {load_path}!")
        return model, optimizer, 0


def evaluate_model(model, test_loader, device):
    model.eval()
    model.to(device)

    margin = 1.5

    running_loss = 0.0
    running_accuracy = 0.0

    with torch.no_grad():
        for audios1, audios2, labels in test_loader:
            audios1 = audios1.float().squeeze(1).to(device)
            audios2 = audios2.float().squeeze(1).to(device)
            labels = labels.to(device)

            embeddings1, embeddings2 = model(audios1, audios2)

            loss = contrastive_loss(embeddings1, embeddings2, labels, margin)

            distances = torch.norm(embeddings1 - embeddings2, dim=1)
            predicts = torch.where(
                distances < margin / 2,
                torch.tensor(1.0, device=device),
                torch.tensor(0.0, device=device),
            )

            accuracy = (predicts == labels).float().mean().item()

            running_loss += loss.item()
            running_accuracy += accuracy

        eval_loss = running_loss / len(test_loader)
        eval_accuracy = running_accuracy / len(test_loader)

        return eval_loss, eval_accuracy


def contrastive_loss(embeddings1, embeddings2, labels, margin=1.0):
    """
    Compute the contrastive loss.

    Parameters:
    - embeddings1, embeddings2: the two sets of embeddings to compare.
    - labels: labels indicating if the pair is similar (1) or dissimilar (0).
    - margin: the margin for dissimilar pairs.

    Returns:
    - loss value
    """
    distances = torch.norm(embeddings1 - embeddings2, dim=1)  # Euclidean distances
    similar_loss = labels * distances**2  # For similar pairs
    dissimilar_loss = (1 - labels) * torch.relu(
        margin - distances
    ) ** 2  # For dissimilar pairs

    return 0.5 * (similar_loss + dissimilar_loss).mean()
