from typing import Any, Tuple
import torch
from torch.utils.data import Dataset
from torchvision.datasets import DatasetFolder, CIFAR10, DTD
from PIL import Image
from collections import Counter
import os

class CustomFolderDataset(Dataset):
    def __init__(self, data_path, transform=None):
        self.new_order = None
        self.transform = transform
        df = DatasetFolder(root=data_path, loader=Image.open, extensions=['.jpg', '.jpeg', '.png'])
        _, class_to_idx = df.find_classes(directory=data_path)

        self.data = DatasetFolder.make_dataset(directory=data_path,class_to_idx=class_to_idx,extensions=['.jpg', '.jpeg', '.png'])  # List of (image_path, label) pairs
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        if self.new_order is not None:
            idx = self.new_order[idx][1]
        image_path, label = self.data[idx]
        image = Image.open(image_path)

        if self.transform:
            image = self.transform(image)

        return image, label, image_path
    
    def reorder(self, new_order):
        self.new_order = new_order

class CustomFileDataset(Dataset):
    def __init__(self, data_path, transform=None):
        # super(CustomFileDataset, self).__init__()
        alphabet = """ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"""
        self.transform = transform
        images = os.listdir(data_path)

        vocabulary = {char: idx for idx, (char) in enumerate(alphabet)}
        # This just combines the relative path with the image name
        f = lambda x: os.path.join(data_path, x)
        self.imagepaths = list(map(f, images))
        self.labels = self.__text_to_tensor(images, vocabulary)
        self.data = list(zip(self.imagepaths, self.labels))
        # Assuming labels is a list of text strings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        assert idx <= len(self), 'index range error'
        imagepath = self.data[idx][0]
        image = Image.open(imagepath)

        if self.transform is not None:
            image = self.transform(image)

        return image, self.data[idx][1], imagepath

    def __text_to_tensor(self, labels, vocabulary):
        converted = []
        for label in labels:
            label = label.split('_')[0]
            converted.append(torch.tensor([vocabulary[char] for char in label], dtype=torch.long))
        return converted
    
class CustomCIFAR10(CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.new_order = None

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if self.new_order is not None:
            index = self.new_order[index][1]
        return super().__getitem__(index)[0], super().__getitem__(index)[1], index

    def reorder(self, new_order):
        self.new_order = new_order

class CustomDTD(DTD):
    def __init__(self, root, split='train', transform=None, target_transform=None, download=False):
        super().__init__(root, split=split, transform=transform, target_transform=target_transform, download=download)
        self.new_order = None

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        if self.new_order is not None:
            index = self.new_order[index][1]
        return super().__getitem__(index)

    def reorder(self, new_order):
        self.new_order = new_order