import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViltModel


class ViltForMultiLabelClassification(nn.Module):
    def __init__(self, num_labels, lock_vilt_weights=False):
        super().__init__()
        self.vilt = ViltModel.from_pretrained("dandelin/vilt-b32-mlm")
        if lock_vilt_weights:
            for param in self.vilt.parameters():
                param.requires_grad = False

        self.classifier = nn.Linear(self.vilt.config.hidden_size, num_labels)

    def forward(self, **kwargs):
        labels = kwargs.pop("labels", None)
        outputs = self.vilt(**kwargs)
        #
        logits = self.classifier(outputs.pooler_output)

        loss = None
        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(logits, labels)

        return logits, loss
