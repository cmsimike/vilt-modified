import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import numpy as np
from sklearn.model_selection import train_test_split
import json
from labels import LabelProcessor
from multi_label_stratified import multi_label_stratified_split


class FoodDataset(Dataset):
    def __init__(self, image_paths, labels, processor, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.processor = processor
        self.transform = transform
        self.prompt_template = "What foods are present in this image?"

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            image = self.transform(image)

        encoding = self.processor(
            images=image,
            text=self.prompt_template,
            return_tensors="pt",
            padding="max_length",
            truncation=True,
        )

        # Remove batch dimension
        for k, v in encoding.items():
            encoding[k] = v.squeeze()

        encoding["labels"] = torch.tensor(self.labels[idx], dtype=torch.float)
        return encoding


def create_dataloaders(config, processor):
    label_processor = LabelProcessor()

    base_directory = os.environ["OUTPUT_DIR_MULTILABEL"]
    json_file = os.path.join(base_directory, "image_to_label.json")

    image_to_label = json.load(open(json_file))

    data = []
    for image_path, label in image_to_label.items():
        tuple_data = (image_path, [x for x in label.split(",")])
        data.append(tuple_data)

    # Split the data
    train_data, test_data, _ = multi_label_stratified_split(
        data,
        test_size=(1 - config.TRAIN_SPLIT),
        random_state=config.RANDOM_SEED,
    )

    # Extract image paths and labels, and process labels for the dataset
    X_train = [item[0] for item in train_data]
    y_train = [label_processor.text_to_tensor(item[1]) for item in train_data]

    X_test = [item[0] for item in test_data]
    y_test = [label_processor.text_to_tensor(item[1]) for item in test_data]

    # Create datasets
    train_dataset = FoodDataset(X_train, y_train, processor)
    test_dataset = FoodDataset(X_test, y_test, processor)

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=4
    )

    test_loader = DataLoader(
        test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4
    )

    return train_loader, test_loader
