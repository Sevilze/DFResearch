import os
import shutil
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
from torch import nn
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_curve, auc, confusion_matrix, precision_recall_curve, average_precision_score, classification_report
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

    def evaluate(self, collect_predictions=False):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0

        all_labels = []
        all_predictions = []
        all_probabilities = []

        pbar = self.init_tqdm(self.test_loader, "Evaluating")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            loss, outputs = self.common_eval_step(inputs, labels)
            total_loss, correct, total = self.update_metrics(
                loss, outputs, labels, total_loss, correct, total
            )

            if collect_predictions:
                probs = torch.nn.functional.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_labels.extend(labels.cpu().numpy())
                all_predictions.extend(preds.cpu().numpy())
                all_probabilities.extend(probs.cpu().detach().numpy())

            pbar.set_postfix(
                {
                    "loss": f"{total_loss / total:.4f}",
                    "acc": f"{100.0 * correct / total:.2f}%",
                    "mem": self.get_memory_usage(),
                }
            )

        if collect_predictions:
            return total_loss / len(self.test_loader), correct / total, {
                "labels": np.array(all_labels),
                "predictions": np.array(all_predictions),
                "probabilities": np.array(all_probabilities)
            }
        else:
            return total_loss / len(self.test_loader), correct / total

    def train(self, epochs):
        best_acc = self.read_best_eval()

        epoch_pbar = tqdm(range(epochs), desc=f"Training {self.model_name}")
        for epoch in epoch_pbar:
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc, metrics_data = self.evaluate(collect_predictions=True)
            self.metrics_data = metrics_data

            self.update_history(train_loss, train_acc, val_loss, val_acc)
            self.update_progress_bar(
                epoch_pbar, train_loss, train_acc, val_loss, val_acc
            )
            self.log_tensorboard(epoch, train_loss, train_acc, val_loss, val_acc)
            self.plot_training_curves(epoch)

            if val_acc > best_acc:
                best_acc = val_acc
                self.update_best_eval(best_acc)
                self.save_scripted_model(epoch, best_acc)
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

    def save_scripted_model(self, epoch, best_acc):
        self.model.eval()
        try:
            scripted_model = torch.jit.script(self.model)
        except Exception as e:
            print(f"Error scripting the model: {e}")
            return
        scripted_model_path = os.path.join(self.best_model_dir, f"{self.model_name}_scripted.pt")
        scripted_model.save(scripted_model_path)
        tqdm.write(f"Saved best model with accuracy {best_acc:.4f} to {scripted_model_path}")

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

        plt.style.use('dark_background')
        sns.set_theme(style="darkgrid", palette="deep")
        colors = sns.color_palette("bright", n_colors=6)
        
        epochs_range = range(1, len(self.history["train_loss"]) + 1)
        fig = plt.figure(figsize=(20, 15), facecolor='#121212')
        gs = gridspec.GridSpec(3, 3, figure=fig)

        text_color = 'white'
        grid_color = '#333333'
        
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(epochs_range, self.history["train_loss"], marker='o', markersize=6, linestyle='-', linewidth=2.0, color=colors[0], label="Train Loss") # Increased marker size
        ax1.plot(epochs_range, self.history["val_loss"], marker='o', markersize=6, linestyle='--', linewidth=2.0, color=colors[1], label="Val Loss") # Increased marker size
        ax1.set_title(f"{self.model_name} Loss", fontsize=14, fontweight='bold', color=text_color)
        ax1.set_xlabel("Epoch", fontsize=12, color=text_color)
        ax1.set_ylabel("Loss", fontsize=12, color=text_color)
        ax1.legend(fontsize=10, facecolor='#1E1E1E', edgecolor='#444444', labelcolor=text_color)
        ax1.tick_params(axis='both', which='major', labelsize=10, colors=text_color)
        ax1.grid(color=grid_color, linestyle='--', linewidth=0.5)
        ax1.set_facecolor('#1E1E1E')

        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(epochs_range, self.history["train_acc"], marker='o', markersize=6, linestyle='-', linewidth=2.0, color=colors[2], label="Train Accuracy") # Increased marker size
        ax2.plot(epochs_range, self.history["val_acc"], marker='o', markersize=6, linestyle='--', linewidth=2.0, color=colors[3], label="Val Accuracy") # Increased marker size
        ax2.set_title(f"{self.model_name} Accuracy", fontsize=14, fontweight='bold', color=text_color)
        ax2.set_xlabel("Epoch", fontsize=12, color=text_color)
        ax2.set_ylabel("Accuracy", fontsize=12, color=text_color)
        ax2.legend(fontsize=10, facecolor='#1E1E1E', edgecolor='#444444', labelcolor=text_color)
        ax2.tick_params(axis='both', which='major', labelsize=10, colors=text_color)
        ax2.grid(color=grid_color, linestyle='--', linewidth=0.5)
        ax2.set_facecolor('#1E1E1E')

        ax3 = fig.add_subplot(gs[0, 2])
        ax3.plot(epochs_range, self.history["lr"], marker='o', markersize=6, linestyle='-', linewidth=2.0, color=colors[4]) # Increased marker size
        ax3.set_title(f"{self.model_name} Learning Rate", fontsize=14, fontweight='bold', color=text_color)
        ax3.set_xlabel("Epoch", fontsize=12, color=text_color)
        ax3.set_ylabel("Learning Rate", fontsize=12, color=text_color)
        ax3.tick_params(axis='both', which='major', labelsize=10, colors=text_color)
        ax3.grid(color=grid_color, linestyle='--', linewidth=0.5)
        ax3.set_facecolor('#1E1E1E')

        if hasattr(self, 'metrics_data') and self.metrics_data is not None:
            ax4 = fig.add_subplot(gs[1, 0])
            cm = confusion_matrix(self.metrics_data['labels'], self.metrics_data['predictions'])
            
            cmap = sns.color_palette("RdPu", as_cmap=True)
            sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, ax=ax4, cbar=True, 
                        annot_kws={"size": 11, "color": "black"},
                        cbar_kws={'label': 'Count', 'shrink': 0.75}) 
                        
            # Set colorbar label color explicitly
            cbar = ax4.collections[0].colorbar
            cbar.ax.yaxis.label.set_color(text_color)
            cbar.ax.tick_params(axis='y', colors=text_color)

            ax4.set_title(f"{self.model_name} Confusion Matrix", fontsize=14, fontweight='bold', color=text_color)
            ax4.set_xlabel('Predicted Label', fontsize=12, color=text_color)
            ax4.set_ylabel('True Label', fontsize=12, color=text_color)
            ax4.tick_params(axis='both', which='major', labelsize=10, colors=text_color)
            ax4.set_facecolor('#1E1E1E')

            ax5 = fig.add_subplot(gs[1, 1])
            n_classes = self.metrics_data['probabilities'].shape[1]

            fpr = dict()
            tpr = dict()
            roc_auc = dict()

            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(
                    (self.metrics_data['labels'] == i).astype(int),
                    self.metrics_data['probabilities'][:, i]
                )
                roc_auc[i] = auc(fpr[i], tpr[i])

            bright_colors = plt.cm.rainbow(np.linspace(0, 1, n_classes))
            
            for i, color in zip(range(n_classes), bright_colors):
                if i in fpr:
                    ax5.plot(
                        fpr[i], tpr[i], color=color, lw=2.5,
                        label=f'Class {i} (AUC = {roc_auc[i]:.3f})' # Label added here
                    )
                    # REMOVED ax5.text(...) to avoid duplicate legend entries

            ax5.plot([0, 1], [0, 1], 'w--', lw=1.5, alpha=0.8)
            ax5.set_xlim([0.0, 1.0])
            ax5.set_ylim([0.0, 1.05])
            ax5.set_xlabel('False Positive Rate', fontsize=12, color=text_color)
            ax5.set_ylabel('True Positive Rate', fontsize=12, color=text_color)
            ax5.set_title(f"{self.model_name} ROC Curve with AUC Scores", fontsize=14, fontweight='bold', color=text_color)
            ax5.legend(loc="lower right", fontsize=9, facecolor='#1E1E1E', edgecolor='#444444', labelcolor=text_color) # Standard legend used
            ax5.tick_params(axis='x', colors=text_color) # Ensure x-axis ticks are visible
            ax5.tick_params(axis='y', colors=text_color) # Ensure y-axis ticks are visible
            ax5.grid(color=grid_color, linestyle='--', linewidth=0.5)
            ax5.set_facecolor('#1E1E1E')

            ax6 = fig.add_subplot(gs[1, 2])
            precision = dict()
            recall = dict()
            avg_precision = dict()

            for i in range(n_classes):
                precision[i], recall[i], _ = precision_recall_curve(
                    (self.metrics_data['labels'] == i).astype(int),
                    self.metrics_data['probabilities'][:, i]
                )
                avg_precision[i] = average_precision_score(
                    (self.metrics_data['labels'] == i).astype(int),
                    self.metrics_data['probabilities'][:, i]
                )

            for i, color in zip(range(n_classes), bright_colors):
                if i in precision:
                    ax6.plot(
                        recall[i], precision[i], color=color, lw=2.5,
                        label=f'Class {i} (AP = {avg_precision[i]:.3f})'
                    )

            ax6.set_xlim([0.0, 1.0])
            ax6.set_ylim([0.0, 1.05])
            ax6.set_xlabel('Recall', fontsize=12, color=text_color)
            ax6.set_ylabel('Precision', fontsize=12, color=text_color)
            ax6.set_title(f"{self.model_name} Precision-Recall Curve", fontsize=14, fontweight='bold', color=text_color)
            ax6.legend(loc="lower left", fontsize=9, facecolor='#1E1E1E', edgecolor='#444444', labelcolor=text_color)
            ax6.tick_params(axis='both', which='major', labelsize=10, colors=text_color)
            ax6.grid(color=grid_color, linestyle='--', linewidth=0.5)
            ax6.set_facecolor('#1E1E1E')

            ax7 = fig.add_subplot(gs[2, 0])
            try:
                report = classification_report(
                    self.metrics_data['labels'],
                    self.metrics_data['predictions'],
                    output_dict=True,
                    zero_division=0
                )
                
                class_indices = [str(i) for i in range(n_classes)]
                metrics = ['precision', 'recall', 'f1-score']
                
                data = []
                for cls in class_indices:
                    for metric in metrics:
                        data.append({
                            'class': f'Class {cls}',
                            'metric': metric,
                            'value': report[cls][metric]
                        })
                
                df = pd.DataFrame(data)
                pivot_df = df.pivot(index='class', columns='metric', values='value')
                pivot_df.plot(kind='bar', ax=ax7, color=bright_colors[:3], width=0.8, alpha=0.9, edgecolor='white', linewidth=0.5)
                
                for container in ax7.containers:
                    ax7.bar_label(container, fmt='%.2f', padding=3, color=text_color, fontweight='bold', fontsize=9)
                
                ax7.set_title(f"{self.model_name} Class Performance Metrics", fontsize=14, fontweight='bold', color=text_color)
                ax7.set_xlabel('Class', fontsize=12, color=text_color)
                ax7.set_ylabel('Score', fontsize=12, color=text_color)
                ax7.legend(title='Metrics', fontsize=10, title_fontsize=11, facecolor='#1E1E1E', edgecolor='#444444', labelcolor=text_color)
                ax7.tick_params(axis='x', rotation=0, labelsize=10, colors=text_color)
                ax7.tick_params(axis='y', labelsize=10, colors=text_color)
                ax7.set_ylim([0, 1.05])  # Standardize y-axis for metrics
                ax7.grid(color=grid_color, linestyle='--', linewidth=0.5, axis='y')
                ax7.set_facecolor('#1E1E1E')
                
            except Exception as e:
                ax7.text(0.5, 0.5, f"Could not generate report:\n{e}", ha='center', va='center', fontsize=11, color=text_color)
                ax7.set_title(f"{self.model_name} Class Metrics", fontsize=14, fontweight='bold', color=text_color)
                ax7.set_facecolor('#1E1E1E')

            ax8 = fig.add_subplot(gs[2, 1])
            pred_counts = np.bincount(self.metrics_data['predictions'], minlength=n_classes)
            true_counts = np.bincount(self.metrics_data['labels'], minlength=n_classes)
            
            labels = [f'Class {i}' for i in range(n_classes)]
            x = np.arange(len(labels))
            width = 0.35
            
            rects1 = ax8.bar(x - width/2, true_counts, width, label='True', color=bright_colors[0], alpha=0.9, edgecolor='white', linewidth=0.5)
            rects2 = ax8.bar(x + width/2, pred_counts, width, label='Predicted', color=bright_colors[1], alpha=0.9, edgecolor='white', linewidth=0.5)
            
            ax8.set_title(f"{self.model_name} Class Distribution", fontsize=14, fontweight='bold', color=text_color)
            ax8.set_xlabel('Class', fontsize=12, color=text_color)
            ax8.set_ylabel('Count', fontsize=12, color=text_color)
            ax8.set_xticks(x)
            ax8.set_xticklabels(labels)
            ax8.legend(fontsize=10, facecolor='#1E1E1E', edgecolor='#444444', labelcolor=text_color)
            ax8.tick_params(axis='both', which='major', labelsize=10, colors=text_color)
            
            ax8.bar_label(rects1, padding=3, color=text_color, fontsize=9)
            ax8.bar_label(rects2, padding=3, color=text_color, fontsize=9)
            
            ax8.grid(color=grid_color, linestyle='--', linewidth=0.5, axis='y')
            ax8.set_facecolor('#1E1E1E')

            ax9 = fig.add_subplot(gs[2, 2])
            for i, color in zip(range(n_classes), bright_colors):
                sns.kdeplot(
                    self.metrics_data['probabilities'][:, i], 
                    label=f'Class {i}', 
                    ax=ax9, 
                    color=color, 
                    fill=True, 
                    alpha=0.4,
                    linewidth=2.5
                )
                
            ax9.set_title(f"{self.model_name} Probability Density", fontsize=14, fontweight='bold', color=text_color)
            ax9.set_xlabel('Probability', fontsize=12, color=text_color)
            ax9.set_ylabel('Density', fontsize=12, color=text_color)
            ax9.legend(fontsize=10, facecolor='#1E1E1E', edgecolor='#444444', labelcolor=text_color)
            ax9.tick_params(axis='both', which='major', labelsize=10, colors=text_color)
            ax9.grid(color=grid_color, linestyle='--', linewidth=0.5)
            ax9.set_facecolor('#1E1E1E')

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])
        fig.suptitle(f"{self.model_name} Training Metrics - Epoch {epoch}", 
                    fontsize=16, fontweight='bold', color=text_color, y=0.99)

        plot_path = os.path.join(self.model_dir, f"{self.model_name}_current_run.png")
        plt.savefig(plot_path, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()

        if os.path.exists(self.best_eval_file):
            best_acc = self.read_best_eval()
            current_val_acc = self.history["val_acc"][-1]
            if current_val_acc >= best_acc:
                best_plot_path = os.path.join(self.best_model_dir, f"{self.model_name}_best_plot.png")
                shutil.copy(plot_path, best_plot_path)
                tqdm.write(f"New best model plot saved to {best_plot_path}")
