import argparse
import torch
import numpy as np
from .dataloader import DataLoaderWrapper
from .loaderconf import BATCH_SIZE, RECOMPUTE_NORM


def parse_args():
    parser = argparse.ArgumentParser(description="Model Training Framework")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["resnet", "densenet"],
        help="Model training options.",
    )
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs.")
    parser.add_argument(
        "--lr", type=float, default=0.001, help="Maximum learning rate value."
    )
    return parser.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    np.random.seed(42)

    data_wrapper = DataLoaderWrapper(BATCH_SIZE, RECOMPUTE_NORM)
    train_loader, test_loader = data_wrapper.get_loaders()

    if args.model == "resnet":
        from resnet.resnetmodel import ResnetClassifier
        from resnet.trainer import ResnetTrainer

        model = ResnetClassifier(
            num_classes=2,
            pretrained=True,
            use_complex_blocks=True,
            in_channels=data_wrapper.input_channels,
        )
        trainer = ResnetTrainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=args.epochs,
            max_lr=args.lr,
        )
    elif args.model == "densenet":
        from densenet.densenetmodel import DensenetClassifier
        from densenet.trainer import DensenetTrainer

        model = DensenetClassifier(
            num_classes=2,
            pretrained=True,
            in_channels=data_wrapper.input_channels,
        )
        trainer = DensenetTrainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=args.epochs,
            max_lr=args.lr,
        )

    trainer.train(trainer.epochs)
    trainer.plot_training_curves()


if __name__ == "__main__":
    main()
