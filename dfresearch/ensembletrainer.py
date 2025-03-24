import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
from tqdm import tqdm
from .dataloader import DataLoaderWrapper
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from resnet.resnetmodel import ResnetClassifier
from .loaderconf import BATCH_SIZE, RECOMPUTE_NORM


class EnsembleTrainer:
    def __init__(self, model, train_loader, test_loader, batch_size, recompute_stats):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.model_name = (
            model.model_name
            if hasattr(model, "model_name")
            else model.__class__.__name__
        )

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)
        print(f"Training {self.model_name} on {self.device}")

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(
            self.model.parameters(), lr=0.001, weight_decay=2e-4
        )
        self.scheduler = None

        self.scaler = torch.amp.grad_scaler.GradScaler(self.device)

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }

    def _get_memory_usage(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.max_memory_allocated() / 1024**2
            return f"{allocated:.2f}/{reserved:.2f} MB"
        return "N/A"

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            self.optimizer.zero_grad()
            with torch.autocast(device_type=self.device.type, dtype=torch.float16):
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            if self.scheduler is not None:
                self.scheduler.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            current_lr = self.optimizer.param_groups[0]["lr"]
            pbar.set_postfix(
                {
                    "loss": f"{total_loss / total:.4f}",
                    "acc": f"{100.0 * correct / total:.2f}%",
                    "lr": f"{current_lr:.2e}",
                    "mem": self._get_memory_usage(),
                }
            )

        return total_loss / len(self.train_loader), correct / total

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad(), torch.autocast(device_type=self.device.type):
            pbar = tqdm(self.test_loader, desc="Evaluating", leave=False)
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                pbar.set_postfix(
                    {
                        "loss": f"{total_loss / total:.4f}",
                        "acc": f"{100.0 * correct / total:.2f}%",
                        "mem": self._get_memory_usage(),
                    }
                )

        return total_loss / len(self.test_loader), correct / total

    def train(self, epochs, max_lr):
        best_acc = 0
        os.makedirs("models", exist_ok=True)

        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=max_lr,
            steps_per_epoch=len(self.train_loader),
            epochs=epochs,
            anneal_strategy="cos",
        )

        epoch_pbar = tqdm(
            range(epochs),
            desc=f"Training {self.model_name}",
            postfix={"best_acc": best_acc},
        )
        for epoch in epoch_pbar:
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.evaluate()

            current_lr = self.optimizer.param_groups[0]["lr"]

            self.history["train_loss"].append(train_loss)
            self.history["train_acc"].append(train_acc)
            self.history["val_loss"].append(val_loss)
            self.history["val_acc"].append(val_acc)
            self.history["lr"].append(current_lr)

            epoch_pbar.set_postfix(
                {
                    "train_loss": f"{train_loss:.4f}",
                    "train_acc": f"{train_acc:.4f}",
                    "val_loss": f"{val_loss:.4f}",
                    "val_acc": f"{val_acc:.4f}",
                    "mem": self._get_memory_usage(),
                }
            )

            if val_acc > best_acc:
                best_acc = val_acc
                save_path = f"models/{self.model_name}/{self.model_name}_best.pth"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scheduler_state_dict": self.scheduler.state_dict(),
                        "best_acc": best_acc,
                    },
                    save_path,
                )
                tqdm.write(f"New best {self.model_name} accuracy: {best_acc:.2%}")

    def plot_training_curves(self):
        plt.figure(figsize=(12, 5))

        plt.subplot(1, 2, 1)
        plt.plot(self.history["train_loss"], label="Train Loss")
        plt.plot(self.history["val_loss"], label="Val Loss")
        plt.title(f"{self.model_name} Loss")
        plt.xlabel("Epoch")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(self.history["train_acc"], label="Train Accuracy")
        plt.plot(self.history["val_acc"], label="Val Accuracy")
        plt.title(f"{self.model_name} Accuracy")
        plt.xlabel("Epoch")
        plt.legend()

        plt.tight_layout()
        plt.savefig(f"models/{self.model_name}/training_{self.model_name}.png")
        plt.close()


def main():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    np.random.seed(42)

    data_wrapper = DataLoaderWrapper(BATCH_SIZE, RECOMPUTE_NORM)
    train_loader, test_loader = data_wrapper.get_loaders()

    models_to_train = [
        ResnetClassifier(
            num_classes=2,
            pretrained=True,
            use_complex_blocks=True,
            in_channels=data_wrapper.input_channels,
        ),
    ]

    for model in models_to_train:
        try:
            print(f"Starting training for {model.__class__.__name__}")
            print(model)

            trainer = EnsembleTrainer(
                model=model,
                train_loader=train_loader,
                test_loader=test_loader,
                batch_size=BATCH_SIZE,
                recompute_stats=RECOMPUTE_NORM,
            )

            trainer.train(epochs=25, max_lr=0.01)
            trainer.plot_training_curves()

        except Exception as e:
            print(f"Error training {model.__class__.__name__}: {str(e)}")
            continue


if __name__ == "__main__":
    main()
