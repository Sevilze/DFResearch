import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
from resnetmodel import get_resnet_model
from torchvision import datasets

class DataLoaderWrapper:
    def __init__(self, data_dir, batch_size=64, val_split=0.2):
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_split = val_split
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        full_dataset = datasets.ImageFolder(root=data_dir, transform=self.transform)
        
        val_size = int(len(full_dataset) * val_split)
        train_size = len(full_dataset) - val_size
        self.train_dataset, self.val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        
        self.train_loader = torch.utils.data.DataLoader(
            self.train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
        self.val_loader = torch.utils.data.DataLoader(
            self.val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        self.class_names = full_dataset.classes

class Trainer:
    def __init__(self, data_dir, num_classes, batch_size=64, lr=0.001):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.data_wrapper = DataLoaderWrapper(data_dir, batch_size)
        self.model = get_resnet_model(num_classes).to(self.device)
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.class_names = self.data_wrapper.class_names
        
    def imshow(self, imgs, labels, title):
        imgs = imgs.cpu().numpy().transpose((0, 2, 3, 1))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        imgs = std * imgs + mean
        imgs = np.clip(imgs, 0, 1)
        
        num_images = min(16, len(imgs))
        grid_size = int(np.sqrt(num_images))
        fig, axes = plt.subplots(grid_size, grid_size, figsize=(10, 10))
        axes = axes.flatten()
        
        for i in range(num_images):
            axes[i].imshow(imgs[i])
            axes[i].set_title(self.class_names[labels[i]])
            axes[i].axis('off')
            
        for j in range(num_images, len(axes)):
            axes[j].axis('off')
            
        plt.suptitle(title)
        plt.show()

    def show_samples(self):
        images, labels = next(iter(self.data_wrapper.train_loader))
        self.imshow(images[:16], labels[:16], "Training Samples")
        
    def train_epoch(self):
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in self.data_wrapper.train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            running_loss += loss.item()
            
        return running_loss/len(self.data_wrapper.train_loader), correct/total
    
    def validate(self):
        self.model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in self.data_wrapper.val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                val_loss += loss.item()
                
        return val_loss/len(self.data_wrapper.val_loader), correct/total
    
    def train(self, epochs=10):
        best_acc = 0.0
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
        
        for epoch in range(epochs):
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)
            
            print(f"Epoch {epoch+1}/{epochs}")
            print(f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}\n")
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(self.model.state_dict(), 'best_model.pth')
                
        self.plot_training(history)
        
    def plot_training(self, history):
        plt.figure(figsize=(12, 5))
        plt.subplot(1, 2, 1)
        plt.plot(history['train_loss'], label='Train Loss')
        plt.plot(history['val_loss'], label='Val Loss')
        plt.legend()
        plt.title('Loss Curves')
        
        plt.subplot(1, 2, 2)
        plt.plot(history['train_acc'], label='Train Acc')
        plt.plot(history['val_acc'], label='Val Acc')
        plt.legend()
        plt.title('Accuracy Curves')
        plt.show()

if __name__ == "__main__":
    trainer = Trainer(
        data_dir='path/to/your/dataset',
        num_classes=10,
        batch_size=64,
        lr=0.001
    )
    trainer.show_samples()
    trainer.train(epochs=15)