import argparse
import os
import matplotlib.pyplot as plt

from datalaoder import EcdcDataLoader
from model import VoiceKeyModel
from torch import optim, nn

import torch

from tqdm import tqdm


def get_parser():
    parser = argparse.ArgumentParser("VoiceKey", description="VoiceKey model")
    parser.add_argument(
        "--source_dir",
        type=str,
        help="Path to the source directory",
        default="data/source",
    )
    parser.add_argument(
        "--label_dir",
        type=str,
        help="Path to the label directory",
        default="data/label",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        help="Batch size",
        default=256,
    )
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=40)
    parser.add_argument("--print_every", type=int, help="Print frequency", default=1)
    parser.add_argument("--num_workers", type=int, help="num_workers", default=24)
    parser.add_argument(
        "--lr_decay_step", type=int, help="Learning rate decay step", default=10
    )
    parser.add_argument(
        "--lr_decay_gamma", type=float, help="Learning rate decay factor", default=0.1
    )

    return parser


def save_model(model, optimizer, epoch, save_dir, model_name):
    """
    Save the model to the path directory provided
    """
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


def main():
    args = get_parser().parse_args()
    batch_size = args.batch_size
    print_every = args.print_every
    loader = EcdcDataLoader(args.source_dir, args.label_dir, batch_size=batch_size)

    criterion = nn.CrossEntropyLoss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = VoiceKeyModel().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma
    )

    # Lists for plotting loss and accuracy
    plot_loss = []
    plot_accuracy = []

    for epoch in range(args.epochs):
        running_loss = 0.0
        running_accuracy = 0.0
        progress_bar = tqdm(enumerate(loader), total=len(loader))

        for i, data in progress_bar:
            # Get the audios; data is a list of [audios, labels]
            audios1, audios2, labels = data
            # Move the audios and the labels to the GPU if available
            audios1 = audios1.float().squeeze(1).to(device)
            audios2 = audios2.float().squeeze(1).to(device)
            labels = labels.to(device)

            # Zero the parameter gradients
            optimizer.zero_grad()

            outputs = model(audios1, audios2)  # shape [batch_size, 2]

            # predictions(according to threshold)
            predictions = torch.argmax(outputs, dim=-1)

            # Compute the accuracy
            accuracy = (predictions == labels).float().mean().item()
            running_accuracy += accuracy

            # Compute the loss
            loss = criterion(outputs, labels)
            # Backward pass
            loss.backward()
            # Optimize
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % print_every == print_every - 1:
                progress_bar.set_postfix(
                    {
                        "epoch": epoch + 1,
                        "loss": running_loss / print_every,
                        "accuracy": accuracy / print_every,
                    }
                )
                plot_loss.append(running_loss / print_every)
                plot_accuracy.append(accuracy / print_every)
                running_loss = 0.0
                running_accuracy = 0.0

        save_model(model, optimizer, epoch, "saved_models", "voicekey")
        scheduler.step()
        print("Epoch [%d] loss: %.6f" % (epoch + 1, running_loss / len(loader)))
    print("Finished Training")

    # Plot the loss and accuracy
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(plot_loss, label="Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(plot_accuracy, label="Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.show()


if __name__ == "__main__":
    main()
