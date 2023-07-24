import argparse
from datalaoder import EcdcDataLoader
from model import EVACModel
from torch import optim, nn

import torch

from tqdm import tqdm
import os
from pathlib import Path


def get_parser():
    parser = argparse.ArgumentParser("EVAC", description="daehwa's EVAC model")
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
        default=512,
    )
    parser.add_argument("--epochs", type=int, help="Number of epochs", default=100)
    parser.add_argument("--print_every", type=int, help="Print frequency", default=2)
    parser.add_argument("--num_workers", type=int, help="num_workers", default=8)

    return parser


def main():
    args = get_parser().parse_args()
    batch_size = args.batch_size
    print_every = args.print_every
    loader = EcdcDataLoader(args.source_dir, args.label_dir, batch_size=batch_size)

    criterion = nn.BCEWithLogitsLoss()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = EVACModel().to(device)
    optimizer = optim.Adam(model.parameters())
    for epoch in range(args.epochs):
        running_loss = 0.0
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

            # Forward pass 일자로 펴주기
            outputs1 = model(audios1).view(audios1.size(0), -1)
            outputs2 = model(audios2).view(audios2.size(0), -1)

            # outputs_shape = [batch_size, 128]
            # cosine similarity
            outputs = torch.cosine_similarity(outputs1, outputs2, dim=-1)

            # 0 ~ 1 사이로 normalize
            outputs = torch.sigmoid(outputs)

            # predictions(according to threshold)
            predictions = torch.where(outputs > 0.5, 1, 0)

            # Compute the accuracy
            accuracy = (predictions == labels).float().mean().item()

            # Compute the loss
            loss = criterion(outputs, labels.float())
            # Backward pass
            loss.backward()
            # Optimize
            optimizer.step()

            # Print statistics
            running_loss += loss.item()
            if i % print_every == print_every - 1:  # print every 2000 mini-batches
                progress_bar.set_postfix(
                    {
                        "epoch": epoch + 1,
                        "loss": running_loss / print_every,
                        "accuracy": accuracy,
                    }
                )
                running_loss = 0.0
        print("Epoch [%d] loss: %.3f" % (epoch + 1, running_loss / len(loader)))
    print("Finished Training")


if __name__ == "__main__":
    main()
