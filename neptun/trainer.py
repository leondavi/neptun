"""Training loop for a DNBN system."""

import time

import torch


class Trainer:

    def __init__(self, system, train_loader, val_loader, config):
        self.system = system
        self.train_loader = train_loader
        self.val_loader = val_loader

        lr = config['training'].get('learning_rate', 0.001)
        weight_decay = config['training'].get('weight_decay', 0.0)
        self.optimizer = torch.optim.Adam(
            system.parameters(), lr=lr, weight_decay=weight_decay
        )
        self.criterion = torch.nn.CrossEntropyLoss()
        self.epochs = config['training'].get('epochs', 10)

        # Optional cosine annealing LR scheduler
        self.scheduler = None
        if config['training'].get('lr_scheduler', '') == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=self.epochs
            )

        # Optional gradient clipping
        self.grad_clip = config['training'].get('grad_clip', 0.0)

    def train(self, device='cpu'):
        """Run full training and return history dict."""
        self.system.to(device)
        history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

        for epoch in range(self.epochs):
            self.system.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            t0 = time.time()

            for batch_idx, (data, target) in enumerate(self.train_loader):
                data, target = data.to(device), target.to(device)

                self.optimizer.zero_grad()
                outputs = self.system(data, step=batch_idx)
                loss, task_loss = self.system.system_loss(
                    outputs, target, self.criterion
                )
                loss.backward()
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.system.parameters(), self.grad_clip
                    )
                self.optimizer.step()

                epoch_loss += task_loss.item()

                # Accuracy from first node
                first_node = next(iter(outputs))
                pred = outputs[first_node].argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

            avg_loss = epoch_loss / len(self.train_loader)
            acc = correct / total
            elapsed = time.time() - t0

            val_loss, val_acc = self._validate(device)

            if self.scheduler is not None:
                self.scheduler.step()

            history['train_loss'].append(avg_loss)
            history['train_acc'].append(acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            current_lr = (
                self.scheduler.get_last_lr()[0]
                if self.scheduler
                else self.optimizer.param_groups[0]['lr']
            )
            print(
                f"Epoch {epoch + 1:3d}/{self.epochs} | "
                f"Train Loss: {avg_loss:.4f}  Acc: {acc:.4f} | "
                f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
                f"LR: {current_lr:.6f} | {elapsed:.1f}s"
            )

        return history

    def _validate(self, device):
        self.system.eval()
        loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in self.val_loader:
                data, target = data.to(device), target.to(device)

                outputs = self.system(data)
                task_loss_sum = sum(
                    self.criterion(out, target) for out in outputs.values()
                )
                loss += (task_loss_sum / len(outputs)).item()

                first_node = next(iter(outputs))
                pred = outputs[first_node].argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)

        return loss / len(self.val_loader), correct / total
