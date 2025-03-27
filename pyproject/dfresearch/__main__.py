import argparse
import torch
import numpy as np
from .dataloader import DataLoaderWrapper
from .loaderconf import BATCH_SIZE, RECOMPUTE_NORM
from ..resnet.resnetmodel import ResnetClassifier
from ..resnet.trainer import ResnetTrainer
from ..densenet.densenetmodel import DensenetClassifier
from ..densenet.trainer import DensenetTrainer
from ..regnet.regnetmodel import RegnetClassifier
from ..regnet.trainer import RegnetTrainer
from .ensemblemodel import EarlyFusionEnsemble
from .ensembletrainer import EnsembleTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Model Training Framework")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["resnet", "densenet", "regnet", "ensemble"],
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

    resnet_model = ResnetClassifier(
        num_classes=2,
        pretrained=True,
        use_complex_blocks=True,
        in_channels=data_wrapper.input_channels,
    )
    densenet_model = DensenetClassifier(
        num_classes=2,
        pretrained=True,
        in_channels=data_wrapper.input_channels,
    )
    regnet_model = RegnetClassifier(
        num_classes=2,
        pretrained=True,
        in_channels=data_wrapper.input_channels,
    )

    if args.model == "resnet":
        model = resnet_model
        trainer = ResnetTrainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=args.epochs,
            max_lr=args.lr,
        )
    elif args.model == "densenet":
        model = densenet_model
        trainer = DensenetTrainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=args.epochs,
            max_lr=args.lr,
        )
    elif args.model == "regnet":

        model = regnet_model
        trainer = RegnetTrainer(
            model=model,
            train_loader=train_loader,
            test_loader=test_loader,
            epochs=args.epochs,
            max_lr=args.lr,
        )
    elif args.model == "ensemble":

        model = EarlyFusionEnsemble(
            num_classes=2,
            in_channels=data_wrapper.input_channels,
            resnet_model=resnet_model,
            densenet_model=densenet_model,
            regnet_model=regnet_model,
            resnet_path="pyproject/models/ResnetClassifier/best_model/ResnetClassifier_best.pth",
            densenet_path="pyproject/models/DensenetClassifier/best_model/DensenetClassifier_best.pth",
            regnet_path="pyproject/models/RegnetClassifier/best_model/RegnetClassifier_best.pth",
            freeze=True
        )
        trainer = EnsembleTrainer(
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
