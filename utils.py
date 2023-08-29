import random
import torch
import numpy as np

import os
import torch
from tqdm import tqdm
import wandb

import matplotlib.pyplot as plt


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


class EarlyStopping:
    def __init__(self, patience=5, verbose=True, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = float("inf")
        self.delta = delta

    def __call__(self, val_loss, model):
        score = -val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f"EarlyStopping counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        """Saves model when validation loss decrease."""
        if self.verbose:
            print(
                f"Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ..."
            )
        torch.save(model.state_dict(), "checkpoint.pt")  # <-- 여기서 모델을 저장
        self.val_loss_min = val_loss


def get_embeddings_from_model(model, test_loader, device):
    model.eval()
    embeddings_list = []
    labels_list = []

    with torch.no_grad():
        for audios1, audios2, labels in test_loader:
            audios1 = audios1.float().squeeze(1).to(device)
            audios2 = audios2.float().squeeze(1).to(device)

            embeddings1, embeddings2 = model(audios1, audios2)
            embeddings_list.append((embeddings1.cpu(), embeddings2.cpu()))
            labels_list.append(labels.cpu())

    return embeddings_list, torch.cat(labels_list)


def plot_roc_curve_for_various_margins(model, test_loader, device, margins):
    embeddings_list, labels = get_embeddings_from_model(model, test_loader, device)
    plt.figure()

    for margin in margins:
        all_distances = []
        for embeddings1, embeddings2 in embeddings_list:
            distances = torch.norm(embeddings1 - embeddings2, dim=1)
            predicts = torch.where(
                distances < margin / 2,
                torch.tensor(1.0),
                torch.tensor(0.0),
            )
            all_distances.append(predicts.numpy())

        distances = np.hstack(all_distances)
        fpr, tpr, _ = roc_curve(labels, distances)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"Margin {margin/2} (area = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic Curve with Varying Margins")
    plt.legend(loc="lower right")
    plt.show()


def calculate_tpr_for_various_margins(model, test_loader, device, margins):
    embeddings_list, labels = get_embeddings_from_model(model, test_loader, device)

    tprs = []

    for margin in margins:
        all_distances = []
        for embeddings1, embeddings2 in embeddings_list:
            distances = torch.norm(embeddings1 - embeddings2, dim=1)
            predicts = torch.where(
                distances < margin / 2,
                torch.tensor(1.0),
                torch.tensor(0.0),
            )
            all_distances.append(predicts.numpy())

        distances = np.hstack(all_distances)
        _, tpr, _ = roc_curve(labels, distances)
        tprs.append(tpr[1])  # TPR 값만 가져옵니다.

    return tprs
