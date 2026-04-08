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
