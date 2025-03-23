import os
import torch
import matplotlib.pyplot as plt
from torch import nn
from tqdm import tqdm

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
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
            "lr": []
        }

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

    def get_save_path(self):
        os.makedirs(f"models/{self.model_name}", exist_ok=True)
        return f"models/{self.model_name}/{self.model_name}_best.pth"

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
            
            pbar.set_postfix({
                "loss": f"{total_loss/total:.4f}",
                "acc": f"{100.*correct/total:.2f}%",
                "mem": self._get_memory_usage()
            })

        return total_loss / len(self.test_loader), correct / total

    def train(self, epochs):
        best_acc = 0

        epoch_pbar = tqdm(range(epochs), desc=f'Training {self.model_name}')
        for epoch in epoch_pbar:
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.evaluate()

            self.update_history(train_loss, train_acc, val_loss, val_acc)
            self.update_progress_bar(epoch_pbar, train_loss, train_acc, val_loss, val_acc)

            if val_acc > best_acc:
                best_acc = val_acc
                self.save_model(epoch, best_acc)

    def update_history(self, train_loss, train_acc, val_loss, val_acc):
        self.history["train_loss"].append(train_loss)
        self.history["train_acc"].append(train_acc)
        self.history["val_loss"].append(val_loss)
        self.history["val_acc"].append(val_acc)
        self.history["lr"].append(self.optimizer.param_groups[0]['lr'])

    def update_progress_bar(self, pbar, train_loss, train_acc, val_loss, val_acc):
        pbar.set_postfix({
            'train_loss': f'{train_loss:.4f}',
            'train_acc': f'{train_acc:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'val_acc': f'{val_acc:.4f}',
            'mem': self._get_memory_usage()
        })

    def save_model(self, epoch, best_acc):
        save_path = self.get_save_path()
        torch.save({
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_acc': best_acc,
        }, save_path)
        tqdm.write(f"New best {self.model_name} accuracy: {best_acc:.2%}")

    def plot_training_curves(self):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.title(f'{self.model_name} Loss')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Accuracy')
        plt.plot(self.history['val_acc'], label='Val Accuracy')
        plt.title(f'{self.model_name} Accuracy')
        plt.xlabel('Epoch')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig(f'models/{self.model_name}/{self.model_name}training_curves.png')
        plt.close()