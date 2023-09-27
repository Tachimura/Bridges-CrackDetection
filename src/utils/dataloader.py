import glob
import os

import numpy as np
import torch.utils.data
import torchvision.transforms as transforms
from PIL import Image
from torch import Generator
from torch.utils.data import Dataset, random_split

train_transformations = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomRotation(360),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomSolarize(0, p=0.5),
        transforms.RandomAdjustSharpness(2, p=0.5),
        transforms.RandomEqualize(p=0.5),
        transforms.RandomAutocontrast(p=0.5),
        transforms.ToTensor(),
    ])

valid_transformations = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])


class CrackDataset(Dataset):
    def __init__(self):
        self.dataset = []

    def __getitem__(self, index):
        img, class_id = self.dataset[index]
        img = Image.open(img)
        return img, class_id

    def __len__(self):
        return len(self.dataset)

    def load_data(self, path, class_id: int = 0):
        cont = 0
        for img in glob.glob(os.path.join(path, "*")):
            self.dataset.append((img, class_id))
            cont += 1
        print(f"Loaded {cont} images with class_id={class_id}")


class ConcreteDatasetWTransforms(Dataset):

    def __init__(self, dataset, transforms):
        self.dataset = dataset
        self.transforms = transforms

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        img, class_id = self.dataset[index]
        img = self.transforms(img)
        return img, class_id


def load_data(path: str, validation_size=0.1, seed=0, batch_size=32):
    dataset = CrackDataset()
    dataset.load_data(os.path.join(path, "Negative"), 0)
    dataset.load_data(os.path.join(path, "Positive"), 1)

    dataset_length = len(dataset)
    valid_length = int(np.floor(validation_size * dataset_length))
    train_length = dataset_length - valid_length
    train_dataset, valid_dataset = random_split(dataset, [train_length, valid_length],
                                                generator=Generator().manual_seed(seed))

    train_dataset = ConcreteDatasetWTransforms(train_dataset, train_transformations)
    valid_dataset = ConcreteDatasetWTransforms(valid_dataset, valid_transformations)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                               num_workers=1, pin_memory=True, persistent_workers=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False,
                                               num_workers=1, pin_memory=True, persistent_workers=True)
    return train_loader, valid_loader
