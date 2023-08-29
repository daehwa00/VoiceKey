import argparse


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
        default=2048,
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
    parser.add_argument("--dim", type=int, help="feature vector dimension", default=128)
    parser.add_argument("--margin", type=float, help="margin", default=1.5)
    parser.add_argument("--dropout", type=float, help="dropout", default=0.1)

    return parser
