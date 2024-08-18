
import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from cv_exp import data_utils


class ImageNetMulti:

    def __init__(self,
                 data_path='./data/imagenet-m') -> None:
        old_classes_map = {}

        with open(os.path.join(os.path.dirname(__file__), 'LOC_synset_mapping.txt')) as f:
            lines = f.readlines()
            for line in lines:
                old_classes_map[len(old_classes_map.keys())] = {
                    'encoded_label': line[:9].strip(),
                    'label': line[9:].split(',')[0].strip(),
                }

            for k in list(old_classes_map.keys()):
                old_classes_map[old_classes_map[k]['encoded_label']] = {
                    'label': old_classes_map[k]['label'].split(',')[0].strip(),
                    'idx': k
                }

        self.mean_t = torch.tensor([0.485, 0.456, 0.406])
        self.std_t = torch.tensor([0.229, 0.224, 0.225])
        self.t = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean_t, std=self.std_t),
        ])

        self.val_dataset = datasets.ImageFolder(
            data_path, transform=self.t)

        self.classes_map = {}

        for k, v in self.val_dataset.class_to_idx.items():
            self.classes_map[v] = {
                'encoded_label': k,
                'label': f"{old_classes_map[k]['idx']}|{old_classes_map[k]['label']}",
            }

            self.classes_map[k] = {
                'label': f"{old_classes_map[k]['idx']}|{old_classes_map[k]['label']}",
                'idx': v,
            }

        self.class_label = np.array(self.val_dataset.imgs)[
            :, 1].astype(np.int32)
