import random
import torch
import numpy as np

import os
import torch
import tqdm
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

    running_loss = 0.0
    running_accuracy = 0.0
    progress_bar = tqdm(enumerate(test_loader), total=len(test_loader))

    for i, data in progress_bar:
        # Get the audios; data is a list of [audios, labels]
        audios1, audios2, labels = data
        # Move the audios and the labels to the GPU if available
        audios1 = audios1.float().squeeze(1).to(device)
        audios2 = audios2.float().squeeze(1).to(device)
        labels = labels.to(device)

        outputs = model(audios1, audios2)
        normlized_outputs = (outputs + 1) / 2

        loss = torch.mean((normlized_outputs - labels.float()) ** 2)

        # Compute the accuracy
        predict = torch.where(normlized_outputs > 0.5, 1, 0)
        accuracy = torch.sum(predict == labels).item() / len(labels)
        wandb.log({"loss": loss.item(), "accuracy": accuracy})
        # statistics
        running_loss += loss.item()
        running_accuracy += accuracy

    eval_loss = running_loss / len(test_loader.dataset)
    eval_accuracy = running_accuracy / len(test_loader.dataset)

    return eval_loss, eval_accuracy
