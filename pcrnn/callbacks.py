from pathlib import Path
import torch

class EarlyStopping:

    def __init__(self, delta=0, patience=5, model_path='model/best_model.pt', restore=True, delete_on_restore=True) -> None:
        self.delta = delta
        self.patience = patience
        self.restore = restore
        self.prev_loss = 0
        self.epoch = 0
        self.patience_count = 0
        self.best_model_epoch = 0
        self.model_path = Path(model_path)
        self.delete_on_restore = delete_on_restore
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

    def evaluate(self, loss, model):
        self.epoch += 1
        if self.epoch == 1:
            self.prev_loss = loss
            self.best_model_epoch = self.epoch
            if self.restore:
                self.store_model(model)
        else:
            if (loss - self.prev_loss) <= self.delta:
                self.prev_loss = loss
                self.patience_count = 0
                self.best_model_epoch = self.epoch
                if self.restore:
                    self.store_model(model)
            else:
                self.patience_count += 1
                if self.patience_count > self.patience:
                    if self.restore:
                        net_checkpoint = self.restore_model()
                        return net_checkpoint
                    else:
                        return None
    
    def store_model(self, model):
        torch.save(model, self.model_path)

    def restore_model(self):
        net = torch.load(self.model_path)
        if self.delete_on_restore:
            self.model_path.unlink()
        return net
