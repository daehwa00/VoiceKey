import torch
from torch import optim, nn
from tqdm import tqdm
import wandb
from datetime import datetime

from dataloader import PreprocessedEcdcDataLoader
from model import VoiceKeyModel, Quant_VoiceKeyModel
from utils import set_seed, save_model, evaluate_model, contrastive_loss
from config import get_parser
import copy
import os

import matplotlib.pyplot as plt

from utils import EarlyStopping

from sklearn.metrics import roc_curve, auc
from matplotlib import font_manager


def train_model(model, train_loader, test_loader, device, args=None):
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=args.lr_decay_step, gamma=args.lr_decay_gamma
    )
    current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # EarlyStopping 객체 초기화
    early_stopping = EarlyStopping(patience=7, verbose=True)

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
            loss = contrastive_loss(embeddings1, embeddings2, labels, args.margin)
            distances = torch.norm(embeddings1 - embeddings2, dim=1)
            predicts = torch.where(
                distances < args.margin / 2,
                torch.tensor(1.0, device=device),
                torch.tensor(0.0, device=device),
            )
            accuracy = (predicts == labels).float().mean().item()

            loss.backward()

            optimizer.step()

            if i == 0:
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
        early_stopping(eval_loss, model)
        if early_stopping.early_stop:
            print("Early stopping")
            break
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
            ["conv3", "bn3", "relu3"],
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
    save_torchscript_model(
        model=quantized_model,
        model_dir="quant_models",
        model_filename="quant_128_adamw",
    )


def initialize_model(dim, device, args):
    model = VoiceKeyModel(dim=dim, args=args).to(device)
    model.train()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total number of parameters: {total_params}")
    return model


def main():
    args = get_parser().parse_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    set_seed(42)

    # WandB – Initialize a new run
    run = wandb.init()
    config = run.config

    train_loader = PreprocessedEcdcDataLoader(
        args.train_dir, batch_size=args.batch_size
    )
    test_loader = PreprocessedEcdcDataLoader(args.test_dir, batch_size=args.batch_size)

    wandb.init(project="voicekey", entity="daehwa")

    # initialize hyperparameters
    args.learning_rate = config.learning_rate
    args.margin = config.margin
    args.dropout = config.dropout

    model = initialize_model(dim=args.dim, device=device, args=args)
    # Add the model and optimizer to wandb
    wandb.watch(model, log="all")

    train_model(
        model=model,
        train_loader=train_loader,
        test_loader=test_loader,
        device=device,
        args=args,
    )


def test():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    args = get_parser().parse_args()
    model = VoiceKeyModel(dim=args.dim, args=args).to(device)

    # Load the checkpoint
    checkpoint_path = "checkpoint.pt"
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        print(f"Model loaded from {checkpoint_path}")
    else:
        print(f"WARNING: No model found in {checkpoint_path}!")
        return

    model.eval()

    test_loader = PreprocessedEcdcDataLoader(args.test_dir, batch_size=args.batch_size)

    all_distances = []
    all_labels = []

    with torch.no_grad():
        for audios1, audios2, labels in test_loader:
            audios1 = audios1.float().squeeze(1).to(device)
            audios2 = audios2.float().squeeze(1).to(device)
            labels = labels.to(device)

            embeddings1, embeddings2 = model(audios1, audios2)

            distances = torch.norm(embeddings1 - embeddings2, dim=1)
            all_distances.extend(distances.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    thresholds = [0.1 * i for i in range(16)]
    plt.figure()
    lw = 2

    results = []

    for threshold in thresholds:
        predicts = [1 if d < threshold else 0 for d in all_distances]
        fpr, tpr, _ = roc_curve(all_labels, predicts)
        roc_auc = auc(fpr, tpr)
        plt.plot(
            fpr,
            tpr,
            lw=lw,
            label=f"ROC curve (Threshold = {threshold}, area = {roc_auc:.2f})",
        )
        results.append((threshold, fpr, tpr))

    plt.plot([0, 1], [0, 1], color="navy", lw=lw, linestyle="--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic Curve for Various Thresholds")
    plt.legend(loc="lower right")
    plt.show()

    # Save the results to a txt file
    with open("roc_results.txt", "w") as f:
        for threshold, fpr, tpr in results:
            f.write(f"Threshold: {threshold}\n")
            f.write(f"False Positive Rate: {fpr[1]}\n")
            f.write(f"True Positive Rate: {tpr[1]}\n")
            f.write("=" * 40 + "\n")


if __name__ == "__main__":
    # main()
    test()


if __name__ == "__main__":
    # main()
    test()
