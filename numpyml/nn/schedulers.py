class LRScheduler:
    def __init__(self, initial_lr, lr_decay_factor=0.1, step_size=10, min_lr=1e-6,
                 patience=1, cooldown=0, patient=True):
        self.initial_lr = initial_lr
        self.lr = initial_lr
        self.lr_decay_factor = lr_decay_factor
        self.step_size = step_size
        self.min_lr = min_lr
        self.patience = patience
        self.cooldown = cooldown
        self.wait = 0
        self.best_loss = float('inf')
        self.cooldown_counter = 0
        self.patient = patient

    def update_lr(self):
        new_lr = max(self.lr * self.lr_decay_factor, self.min_lr)
        print(f"Learning rate scheduler: new_lr = {new_lr}, current_lr = {self.lr}")
        if new_lr < self.lr:
            self.lr = new_lr
            self.cooldown_counter = self.cooldown
            self.wait = 0

    def on_epoch_end(self, epoch, val_loss):
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            self.wait = 0
        if self.patient:
            if val_loss < self.best_loss:
                self.best_loss = val_loss
                self.wait = 0
            else:
                self.wait += 1
            if self.wait >= self.patience:
                self.update_lr()
        else:
            if (epoch + 1) % self.step_size == 0:
                self.update_lr()

    def __call__(self, epoch, val_loss=None):
        if val_loss is not None:
            self.on_epoch_end(epoch, val_loss)
        return self.lr


class CosineAnnealingLR:
    def __init__(self, initial_lr, T_max, min_lr=0.0):
        self.initial_lr = initial_lr
        self.lr = initial_lr
        self.T_max = T_max
        self.min_lr = min_lr

    def __call__(self, epoch, val_loss=None):
        import math
        self.lr = self.min_lr + 0.5 * (self.initial_lr - self.min_lr) * (1 + math.cos(math.pi * epoch / self.T_max))
        return self.lr


class WarmupScheduler:
    def __init__(self, scheduler, warmup_epochs, warmup_start_lr=1e-7):
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.warmup_start_lr = warmup_start_lr
        self.initial_lr = scheduler.initial_lr
        self.lr = warmup_start_lr

    def __call__(self, epoch, val_loss=None):
        if epoch < self.warmup_epochs:
            self.lr = self.warmup_start_lr + (self.initial_lr - self.warmup_start_lr) * epoch / self.warmup_epochs
        else:
            self.lr = self.scheduler(epoch - self.warmup_epochs, val_loss)
        return self.lr


class ExponentialLR:
    def __init__(self, initial_lr, decay_rate=0.95, min_lr=1e-7):
        self.initial_lr = initial_lr
        self.lr = initial_lr
        self.decay_rate = decay_rate
        self.min_lr = min_lr

    def __call__(self, epoch, val_loss=None):
        self.lr = max(self.initial_lr * (self.decay_rate ** epoch), self.min_lr)
        return self.lr
