import json
import numpy as np
from pathlib import Path
import os
from categories import get_categories


# Class to help convert text labels to tensors and viceversa
class LabelProcessor(object):
    def __init__(self):
        self.categories = get_categories()
        self.label_to_idx = {label: idx for idx, label in enumerate(self.categories)}
        self.idx_to_label = {idx: label for idx, label, in enumerate(self.categories)}

    def text_to_tensor(self, text_labels):
        if isinstance(text_labels, str):
            text_labels = [l.strip().lower() for l in text_labels.split(",")]

        tensor = np.zeros(len(self.label_to_idx))
        for label in text_labels:
            label = label.strip().lower()
            if label in self.label_to_idx:
                tensor[self.label_to_idx[label]] = 1
        return tensor

    # TODO is this being used?
    def tensor_to_text(self, tensor, threshold=0.5):
        active_indices = np.where(tensor > threshold)[0]
        return [self.idx_to_label[idx] for idx in active_indices]

    def get_categories(self):
        return self.categories
