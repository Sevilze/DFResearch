import torch
from ..dfresearch.basetrainer import BaseTrainer


class RegnetTrainer(BaseTrainer):
    def __init__(self, model, train_loader, test_loader, epochs, max_lr):
        super().__init__(model, train_loader, test_loader, epochs, max_lr)

        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=0.001,
            weight_decay=1e-4,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=max_lr,
            steps_per_epoch=len(self.train_loader),
            epochs=epochs,
            anneal_strategy="cos",
            pct_start=0.3,
            div_factor=10,
            final_div_factor=1e4,
        )

        self.scaler = torch.amp.grad_scaler.GradScaler(
            init_scale=2.0**16,
            growth_factor=2.0,
            backoff_factor=0.5,
            growth_interval=2000,
        )

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0

        pbar = self.init_tqdm(self.train_loader, "Training")
        for inputs, labels in pbar:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            loss, outputs = self.common_train_step(inputs, labels)

            if self.scheduler:
                self.scheduler.step()

            total_loss, correct, total = self.update_metrics(
                loss, outputs, labels, total_loss, correct, total
            )

            pbar.set_postfix(
                {
                    "loss": f"{total_loss / total:.4f}",
                    "acc": f"{100.0 * correct / total:.2f}%",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.2e}",
                    "mem": self.get_memory_usage(),
                }
            )

        return total_loss / len(self.train_loader), correct / total
