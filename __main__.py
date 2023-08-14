import argparse
import os

from datalaoder import PreprocessedEcdcDataLoader
from model import VoiceKeyModel, VoiceKeyModelWithConv
from torch import optim, nn
from utils import set_seed, save_model
from datetime import datetime
import torch

from tqdm import tqdm
import wandb


def get_parser():
    parser = argparse.ArgumentParser("VoiceKey", description="VoiceKey model")
    parser.add_argument(
        "--train_dir",
        type=str,
        help="Path to the source directory",
        default="saved_data/preprocessed_train_data.pth",
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        help="Path to the label directory",
        default="saved_data/preprocessed_test_data.pth",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size",
        default=4096,
    )
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=100)
    parser.add_argument("--print_every", type=int, help="Print frequency", default=2)
    parser.add_argument("--num_workers", type=int, help="num_workers", default=24)
    parser.add_argument(
        "--lr_decay_step", type=int, help="Learning rate decay step", default=20
    )
    parser.add_argument(
        "--lr_decay_gamma", type=float, help="Learning rate decay factor", default=0.5
    )
    parser.add_argument(
        "--learning_rate", type=float, help="Learning rate", default=2e-3
    )

    return parser


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


def train_model(model, train_loader, test_loader, device, args=None):
    # The training configurations were not carefully selected.

    criterion = nn.CrossEntropyLoss()

    model.to(device)

    # It seems that SGD optimizer is better than Adam optimizer for ResNet18 training on CIFAR10.
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma
    )
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for epoch in range(args.epochs):
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))

        for i, data in progress_bar:
            # Get the audios; data is a list of [audios, labels]
            audios1, audios2, labels = data
            # Move the audios and the labels to the GPU if available
            audios1 = audios1.float().squeeze(1).to(device)
            audios2 = audios2.float().squeeze(1).to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(audios1, audios2)
            normlized_outputs = (outputs + 1) / 2

            loss = torch.mean((normlized_outputs - labels.float()) ** 2)

            # Compute the accuracy
            predict = torch.where(normlized_outputs > 0.5, 1, 0)
            accuracy = torch.sum(predict == labels).item() / len(labels)

            # Backward pass
            loss.backward()

            # Optimize
            optimizer.step()

            wandb.log({"loss": loss.item(), "accuracy": accuracy})

        save_model(model, optimizer, epoch, "saved_models", "voicekey", current_time)
        scheduler.step()
        print(
            "Epoch [%d] loss: %.6f accuracy: %.3f"
            % (epoch + 1, loss.item(), accuracy * 100)
        )

    # Evaluation
    model.eval()
    eval_loss, eval_accuracy = evaluate_model(
        model=model, test_loader=test_loader, device=device, criterion=criterion
    )
    print(
        "Epoch: {:02d} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(
            -1, eval_loss, eval_accuracy
        )
    )
    return model


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


def main():
    cuda_device = torch.device("cuda:0")
    args = get_parser().parse_args()
    # loader = EcdcDataLoader(args.source_dir, args.label_dir, batch_size=batch_size)

    train_loader = PreprocessedEcdcDataLoader(
        args.train_dir, batch_size=args.batch_size
    )
    test_loader = PreprocessedEcdcDataLoader(args.test_dir, batch_size=args.batch_size)
    set_seed(42)

    wandb.init(project="voicekey", entity="daehwa")
    dim = 64
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = VoiceKeyModel(dim=dim).to(device)
    model.train()

    # model, optimizer, start_epoch = load_model(model, optimizer, "saved_models", "voicekey_78")

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")

    # Add the model and optimizer to wandb
    wandb.watch(model, log="all")

    train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=cuda_device,
        args=args,
    )


if __name__ == "__main__":
    main()
