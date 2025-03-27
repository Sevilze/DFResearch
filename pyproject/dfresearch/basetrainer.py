import os
import shutil
import torch
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

class BaseTrainer:
    def __init__(self, model, train_loader, test_loader, epochs, max_lr):
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.model = model
        self.epochs = epochs
        self.max_lr = max_lr
        self.model_name = model.__class__.__name__

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.model.to(self.device)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = None
        self.scheduler = None
        self.scaler = None

        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "lr": [],
        }

        self.model_dir = os.path.join("pyproject/models", self.model_name)
        os.makedirs(self.model_dir, exist_ok=True)
        self.runs_dir = os.path.join(self.model_dir, "runs")
        if os.path.exists(self.runs_dir):
            shutil.rmtree(self.runs_dir)
        os.makedirs(self.runs_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.runs_dir)
        self.best_model_dir = os.path.join(self.model_dir, "best_model")
        os.makedirs(self.best_model_dir, exist_ok=True)
        self.best_eval_file = os.path.join(self.best_model_dir, "best_eval.txt")

    def get_memory_usage(self):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.max_memory_allocated() / 1024**2
            return f"{allocated:.2f}/{reserved:.2f} MB"
        return "N/A"

    def init_tqdm(self, data_loader, desc):
        return tqdm(data_loader, desc=desc, leave=False)

    def common_train_step(self, inputs, labels):
        self.optimizer.zero_grad()
        with torch.autocast(device_type=self.device.type, dtype=torch.float16):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()
        return loss, outputs

    def common_eval_step(self, inputs, labels):
        with torch.no_grad(), torch.autocast(device_type=self.device.type):
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
        return loss, outputs

    def update_metrics(self, loss, outputs, labels, total_loss, correct, total):
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        return total_loss, correct, total

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        pbar = self.init_tqdm(self.test_loader, "Evaluating")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            loss, outputs = self.common_eval_step(inputs, labels)
            total_loss, correct, total = self.update_metrics(
                loss, outputs, labels, total_loss, correct, total
            )

            pbar.set_postfix(
                {
                    "loss": f"{total_loss / total:.4f}",
                    "acc": f"{100.0 * correct / total:.2f}%",
                    "mem": self.get_memory_usage(),
                }
            )

        return total_loss / len(self.test_loader), correct / total

    def train(self, epochs):
        best_acc = self.read_best_eval()

        epoch_pbar = tqdm(range(epochs), desc=f"Training {self.model_name}")
        for epoch in epoch_pbar:
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.evaluate()

            self.update_history(train_loss, train_acc, val_loss, val_acc)
            self.update_progress_bar(
                epoch_pbar, train_loss, train_acc, val_loss, val_acc
            )
            self.log_tensorboard(epoch, train_loss, train_acc, val_loss, val_acc)
            self.plot_training_curves(epoch)

            if val_acc > best_acc:
                best_acc = val_acc
                self.update_best_eval(best_acc)
                self.save_model(epoch, best_acc)
                self.copy_tfoutput()

    def update_history(self, train_loss, train_acc, val_loss, val_acc):
        self.history["train_loss"].append(train_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)
        self.history["lr"].append(self.optimizer.param_groups[0]["lr"])

    def update_progress_bar(self, pbar, train_loss, train_acc, val_loss, val_acc):
        pbar.set_postfix(
            {
                "train_loss": f"{train_loss:.4f}",
                "train_acc": f"{train_acc:.4f}",
                "val_loss": f"{val_loss:.4f}",
                "val_acc": f"{val_acc:.4f}",
                "mem": self.get_memory_usage(),
            }
        )

    def log_tensorboard(self, epoch, train_loss, train_acc, val_loss, val_acc):
        self.writer.add_scalar("Loss/Train", train_loss, epoch)
        self.writer.add_scalar("Loss/Validation", val_loss, epoch)
        self.writer.add_scalar("Accuracy/Train", train_acc, epoch)
        self.writer.add_scalar("Accuracy/Validation", val_acc, epoch)
        self.writer.add_scalar(
            "LearningRate", self.optimizer.param_groups[0]["lr"], epoch
        )
        self.writer.flush()

    def save_model(self, epoch, best_acc):
        save_path = os.path.join(self.best_model_dir, f"{self.model_name}_best.pth")
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
        tqdm.write(
            f"New best {self.model_name} accuracy: {best_acc:.2%} (saved to {save_path})"
        )

    def read_best_eval(self):
        if os.path.exists(self.best_eval_file):
            try:
                with open(self.best_eval_file, "r") as f:
                    best_acc = float(f.read().strip())
                    return best_acc
            except Exception:
                return 0
        return 0

    def update_best_eval(self, best_acc):
        with open(self.best_eval_file, "w") as f:
            f.write(f"{best_acc}")

    def copy_tfoutput(self):
        for file in os.listdir(self.runs_dir):
            if file.startswith("events.out.tfevents"):
                src = os.path.join(self.runs_dir, file)
                dst = os.path.join(self.best_model_dir, "tfoutput_" + file)
                shutil.copy(src, dst)
                tqdm.write(f"Copied TensorBoard event file to {dst}")
                break

    def plot_training_curves(self, epoch):
        if epoch == 0:
            return
        
        epochs_range = range(1, len(self.history["train_loss"]) + 1)
        
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(epochs_range, self.history["train_loss"], label="Train Loss")
        plt.plot(epochs_range, self.history["val_loss"], label="Val Loss")
        plt.title(f"{self.model_name} Loss")
        plt.xlabel("Epoch")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(epochs_range, self.history["train_acc"], label="Train Accuracy")
        plt.plot(epochs_range, self.history["val_acc"], label="Val Accuracy")
        plt.title(f"{self.model_name} Accuracy")
        plt.xlabel("Epoch")
        plt.legend()

        plt.tight_layout()
        plot_path = os.path.join(self.model_dir, f"{self.model_name}_current_run.png")
        plt.savefig(plot_path)
        plt.close()

        if os.path.exists(self.best_eval_file):
            best_acc = self.read_best_eval()
            current_val_acc = self.history["val_acc"][-1]
            if current_val_acc >= best_acc:
                best_plot_path = os.path.join(self.best_model_dir, f"{self.model_name}_best_plot.png")
                shutil.copy(plot_path, best_plot_path)
                tqdm.write(f"Copied plot to {best_plot_path}")
