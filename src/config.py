import os
from categories import get_category_count


class Config(object):
    def __init__(
        self,
        batch_size=8,
        num_epochs=5,
        learning_rate=5e-5,
        train_split=0.8,
        random_seed=42,
        lock_vilt_weights=False,
        false_threshold=0.5,
    ):
        self.BATCH_SIZE = batch_size
        self.NUM_EPOCHS = num_epochs
        self.LEARNING_RATE = learning_rate
        self.NUM_LABELS = get_category_count()
        self.TRAIN_SPLIT = train_split
        self.RANDOM_SEED = random_seed
        self.LOCK_VILT_WEIGHTS = lock_vilt_weights
        self.FALSE_THRESHOLD = false_threshold

    def __str__(self):
        return (
            f"Configuration:\n"
            f"  Batch Size: {self.BATCH_SIZE}\n"
            f"  Number of Epochs: {self.NUM_EPOCHS}\n"
            f"  Learning Rate: {self.LEARNING_RATE}\n"
            f"  Train Split: {self.TRAIN_SPLIT}\n"
            f"  Random Seed: {self.RANDOM_SEED}\n"
            f"  Lock ViLT Weights: {self.LOCK_VILT_WEIGHTS}\n"
            f"  False Threshold: {self.FALSE_THRESHOLD}\n"
        )

    def to_dict(self):
        return {
            "BATCH_SIZE": self.BATCH_SIZE,
            "NUM_EPOCHS": self.NUM_EPOCHS,
            "LEARNING_RATE": self.LEARNING_RATE,
            "TRAIN_SPLIT": self.TRAIN_SPLIT,
            "RANDOM_SEED": self.RANDOM_SEED,
            "LOCK_VILT_WEIGHTS": self.LOCK_VILT_WEIGHTS,
            "FALSE_THRESHOLD": self.FALSE_THRESHOLD,
        }
