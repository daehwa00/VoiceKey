import torch
from torch import optim, nn
from tqdm import tqdm
import wandb
from datetime import datetime

from dataloader import PreprocessedEcdcDataLoader
from model import VoiceKeyModel
from utils import set_seed, save_model, evaluate_model
from config import get_parser


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


def train_model(model, train_loader, test_loader, device, args=None):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma
    )
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    margin = 1.5

    for epoch in range(args.epochs):
        progress_bar = tqdm(enumerate(train_loader), total=len(train_loader))

        for i, data in progress_bar:
            # Get the audios; data is a list of [audios, labels]
            audios1, audios2, labels = data
            # Move the audios and the labels to the GPU if available
            audios1 = audios1.float().squeeze(1).to(device)
            audios2 = audios2.float().squeeze(1).to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            embeddings1, embeddings2 = model(audios1, audios2)  # -1 ~ 1
            # normlized_outputs = (outputs + 1) / 2  # 0 ~ 1

            # loss = torch.mean((normlized_outputs - labels.float()) ** 2)
            # loss = criterion(outputs, labels.float())
            loss = contrastive_loss(embeddings1, embeddings2, labels, margin)
            distances = torch.norm(embeddings1 - embeddings2, dim=1)
            predicts = torch.where(
                distances < margin / 2,
                torch.tensor(1.0, device=device),
                torch.tensor(0.0, device=device),
            )
            accuracy = (predicts == labels).float().mean().item()
            # Compute the accuracy
            # predict = torch.where(normlized_outputs > 0.5, 1, 0)
            # accuracy = torch.sum(predict == labels).item() / len(labels)

            loss.backward()

            optimizer.step()

            if i % 4 == 0:
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
        model=model, test_loader=test_loader, device=device
    )
    print(
        "Epoch: {:02d} Eval Loss: {:.3f} Eval Acc: {:.3f}".format(
            -1, eval_loss, eval_accuracy
        )
    )
    return model


def initialize_model(dim, device):
    model = VoiceKeyModel(dim=dim).to(device)
    model.train()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    return model


def main():
    args = get_parser().parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(42)
    train_loader = PreprocessedEcdcDataLoader(
        args.train_dir, batch_size=args.batch_size
    )
    test_loader = PreprocessedEcdcDataLoader(args.test_dir, batch_size=args.batch_size)

    wandb.init(project="voicekey", entity="daehwa")
    model = initialize_model(dim=args.dim, device=device)

    # Add the model and optimizer to wandb
    wandb.watch(model, log="all")

    train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        args=args,
    )


if __name__ == "__main__":
    main()
