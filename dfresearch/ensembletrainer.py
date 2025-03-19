import torch
import matplotlib.pyplot as plt
import numpy as np
from torch import nn, optim
from tqdm import tqdm
from dataloader import DataLoaderWrapper
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from resnet.resnetmodel import ret_resnet
from loaderconf import BATCH_SIZE, VAL_SPLIT

class EnsembleTrainer:
    def __init__(self, batch_size, val_split):
        self.data_wrapper = DataLoaderWrapper(batch_size, val_split)
        self.train_loader, self.test_loader = self.data_wrapper.get_loaders()
        self.num_classes = len(self.data_wrapper.train_loader.dataset.classes)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        self.model = ret_resnet(num_classes=self.num_classes)
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=1e-4)
        
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc='Training', leave=False)
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{total_loss/total:.4f}',
                'acc': f'{100.*correct/total:.2f}%'
            })
        
        return total_loss / len(self.train_loader), correct / total

    def evaluate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(self.test_loader, desc='Evaluating', leave=False)
            for inputs, labels in pbar:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{total_loss/total:.4f}',
                    'acc': f'{100.*correct/total:.2f}%'
                })
        
        return total_loss / len(self.test_loader), correct / total

    def train(self, epochs=25):
        best_acc = 0
        
        os.makedirs('models', exist_ok=True)
        
        epoch_pbar = tqdm(range(epochs), desc='Epochs')
        for epoch in epoch_pbar:
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.evaluate()
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            
            epoch_pbar.set_postfix({
                'train_loss': f'{train_loss:.4f}',
                'train_acc': f'{train_acc:.4f}',
                'val_loss': f'{val_loss:.4f}',
                'val_acc': f'{val_acc:.4f}'
            })
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'best_acc': best_acc,
                }, 'models/best_model.pth')
            
            if (epoch + 1) % 5 == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'history': self.history,
                }, f'models/checkpoint_epoch_{epoch+1}.pth')

    def plot_training_curves(self):
        plt.figure(figsize=(12, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.history['train_loss'], label='Train Loss')
        plt.plot(self.history['val_loss'], label='Val Loss')
        plt.title('Training and Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(self.history['train_acc'], label='Train Accuracy')
        plt.plot(self.history['val_acc'], label='Val Accuracy')
        plt.title('Training and Validation Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig('training_curves.png')
        plt.show()

def main():
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)
    np.random.seed(42)

    trainer = EnsembleTrainer(BATCH_SIZE, VAL_SPLIT)
    
    try:
        trainer.train(epochs=25)
        trainer.plot_training_curves()
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        torch.save({
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'history': trainer.history,
        }, 'models/interrupted_training.pth')
        
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise e

if __name__ == "__main__":
    main()
