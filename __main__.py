import torch
from torch import optim, nn
from tqdm import tqdm
import wandb
from datetime import datetime

from dataloader import PreprocessedEcdcDataLoader
from model import VoiceKeyModel, Quant_VoiceKeyModel
from utils import set_seed, save_model, evaluate_model
from config import get_parser
import copy
import os


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


def save_torchscript_model(model, model_dir, model_filename):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_filepath = os.path.join(model_dir, model_filename)
    torch.jit.save(torch.jit.script(model), model_filepath)


def load_torchscript_model(model_filepath, device):
    model = torch.jit.load(model_filepath, map_location=device)

    return model


def quantization():
    cuda_device = torch.device("cuda:0")
    cpu_device = torch.device("cpu:0")
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = get_parser().parse_args()
    batch_size = args.batch_size

    model = Quant_VoiceKeyModel()
    wandb.init(project="voicekey", entity="daehwa")
    wandb.watch(model, log="all")

    model.to(cpu_device)
    fused_model = copy.deepcopy(model)
    model.eval()
    fused_model.eval()
    quantization_config = torch.quantization.get_default_qconfig("fbgemm")
    fused_model.qconfig = quantization_config
    fused_model = torch.quantization.fuse_modules(
    fused_model,
    [
        ["conv1", "bn1", "relu1"],
        ["conv2", "bn2", "relu2"],
        ["conv3", "bn3", "relu3"]
    ],
    inplace=True,
    )

    print("Training QAT Model...")
    fused_model.train()
    torch.quantization.prepare_qat(fused_model, inplace=True)

    train_data_path = "saved_data/preprocessed_train_data.pth"
    train_loader = PreprocessedEcdcDataLoader(train_data_path, batch_size=batch_size)

    test_data_path = "saved_data/preprocessed_test_data.pth"
    test_loader = PreprocessedEcdcDataLoader(test_data_path, batch_size=batch_size)
    set_seed(42)

    
    train_model(
        model=fused_model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=cuda_device,
        args=args,
    )

    fused_model.to(cpu_device).eval()
    quantized_model = torch.quantization.convert(fused_model, inplace=True)
    print(quantized_model)
    save_torchscript_model(model=quantized_model, model_dir="quant_models", model_filename="quant_128_adamw")


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
    # main()
    quantization()
