import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import pathlib

current_path = pathlib.Path(__file__).parent.resolve()


class Places365:

    def __init__(self,
                 data_path='./data/places365/val',
                 label_path='./data/places365/val.txt',
                 label_map_path=os.path.join(current_path, 'categories_places365.txt')) -> None:
        classes_map = {}

        self.mean_t = torch.tensor([0.485, 0.456, 0.406])
        self.std_t = torch.tensor([0.229, 0.224, 0.225])
        self.st = transforms.Compose([
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean_t, std=self.std_t),
        ])

        with open(os.path.join(label_map_path)) as f:
            lines = f.readlines()
            for line in lines:
                encoded, label_idx = line.split(' ')
                classes_map[int(label_idx)] = {
                    'encoded_label': encoded.strip(),
                    'label': encoded.split('/')[2].strip(),
                }
        self.classes_map = classes_map

        self.val_idx_path = os.path.join(current_path, 'valid_val.npy')
        if os.path.exists(self.val_idx_path):
            with open(self.val_idx_path, 'rb') as f:
                valid_val = np.load(f)
            self.val_dataset = torch.utils.data.Subset(
                datasets.ImageFolder(data_path, transform=self.st), valid_val)
            self.class_label = np.array(self.val_dataset.dataset.imgs)[
                :, 1].astype(np.int32)[valid_val]
        else:
            self.val_dataset = datasets.ImageFolder(
                data_path, transform=self.st)

            self.class_label = np.array(self.val_dataset.imgs)[
                :, 1].astype(np.int32)
