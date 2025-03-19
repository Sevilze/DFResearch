import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
from resnetmodel import ret_resnet

class Dataloader:
    def __init__(self, data_dir, batch_size=64, val_split=0.2):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split
        
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(30),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            transforms.RandomErasing(p=0.5, scale=(0.02, 0.33)), 
        ])
        
        self.val_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        full_dataset = datasets.ImageFolder(root=data_dir)
        
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        
        train_dataset.dataset.transform = self.train_transform
        val_dataset.dataset.transform = self.val_transform
        
        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
        self.val_loader = torch.utils.data.DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
        
        self.class_names = full_dataset.classes

class Trainer:
    def __init__(self, data_dir, num_classes, batch_size=64, lr=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_loader = Dataloader(data_dir, batch_size)
        self.model = ret_resnet(num_classes).to(self.device)
        self._setup_training(lr)
        self.class_names = self.data_loader.class_names
        
    def _setup_training(self, lr):
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.optimizer = optim.AdamW(self.model.parameters(), lr=lr, weight_decay=1e-4)
        self.scheduler = OneCycleLR(
            self.optimizer, 
            max_lr=lr*10,
            total_steps=None,
            epochs=25,
            steps_per_epoch=len(self.data_loader.train_loader),
            pct_start=0.3,
            anneal_strategy='cos'
        )
        self.scaler = torch.cuda.amp.GradScaler()
        
    def _train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in self.data_loader.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            
            with torch.cuda.amp.autocast():
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            running_loss += loss.item()
            
        return running_loss/len(self.data_loader.train_loader), correct/total
    
    def _validate(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.data_loader.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                val_loss += loss.item()
                
        return val_loss/len(self.data_loader.val_loader), correct/total
    
    def train(self, epochs=25):
        best_acc = 0.0
        history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'lr': []
        }
        
        for epoch in range(epochs):
            train_loss, train_acc = self._train_epoch()
            val_loss, val_acc = self._validate()
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            history['lr'].append(self.scheduler.get_last_lr()[0])
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}%")
            print(f"Val Loss: {val_loss:.4f} | Acc: {val_acc*100:.2f}%")
            print(f"LR: {history['lr'][-1]:.6f}\n")
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'scheduler_state': self.scheduler.state_dict(),
                }, 'best_model.pth')
                
        self._plot_training(history)
        
    def _plot_training(self, history):
        plt.figure(figsize=(18, 6))
        
        plt.subplot(1, 3, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.title('Loss Curves')
        plt.legend()
        
        plt.subplot(1, 3, 2)
        plt.plot(history['train_acc'], label='Train Acc')
        plt.plot(history['val_acc'], label='Val Acc')
        plt.title('Accuracy Curves')
        plt.legend()
        
        plt.subplot(1, 3, 3)
        plt.plot(history['lr'], label='Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.legend()
        
        plt.tight_layout()
        plt.show()