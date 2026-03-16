import numpy as np

SEED_DICT = {
    'Almond':      42,
    'Pistachio':   42,
    'GarlicStems': 42,
}


class UniversalEarlyStopping:
    """
    Monitors validation reconstruction loss to prevent Identity Mapping Problem.
    
    Trains until the model stops meaningfully improving on the spatially isolated
    Validation Block, ensuring fair evaluation across architectures.
    """
    def __init__(self, patience=50, min_delta=1e-4):
        """
        Args:
            patience (int): How many evaluations to wait after the last improvement.
            min_delta (float): Minimum absolute change to qualify as improvement.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = np.inf
        self.early_stop = False

    def __call__(self, val_loss):
        """Check if training should stop."""
        if self.best_loss - val_loss > self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter % 10 == 0:  # Print every 10 patience steps
                print(f"  [EarlyStopping] Patience {self.counter}/{self.patience} (Best: {self.best_loss:.5f})")
            if self.counter >= self.patience:
                self.early_stop = True
