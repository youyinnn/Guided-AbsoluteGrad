import os
import sys
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from cv_exp.utils import data_utils
import pathlib
from torch.utils.data import random_split
import random

data_path = "./data/imagenet-s/validation"
seg_path = "./data/imagenet-s/validation-segmentation"


class ImageNetSeg:

    def __init__(
        self,
        data_path=data_path,
        seg_path=seg_path,
        use_seg_sal=True,
        use_valid_val=True,
        size=(256, 256),
    ) -> None:
        classes_map = {}

        self.data_path = data_path
        self.seg_path = seg_path

        if not os.path.exists(f"{os.path.abspath(seg_path)}-sal"):
            print("seg sal not exist")
            use_seg_sal = False
        else:
            if use_seg_sal:
                print("seg sal is used")
                self.seg_path = f"{os.path.abspath(seg_path)}-sal"
            else:
                print("seg sal not exist")

        current_path = pathlib.Path(__file__).parent.resolve()

        with open(os.path.join(current_path, "LOC_synset_mapping.txt")) as f:
            lines = f.readlines()
            for line in lines:
                classes_map[len(classes_map.keys())] = {
                    "encoded_label": line[:9].strip(),
                    "label": line[9:].split(",")[0].strip(),
                }

            for k in list(classes_map.keys()):
                classes_map[classes_map[k]["encoded_label"]] = {
                    "label": classes_map[k]["label"].split(",")[0].strip(),
                    "idx": k,
                }

        self.classes_map = classes_map
        self.use_valid_val = use_valid_val

        self.mean_t = torch.tensor([0.485, 0.456, 0.406])
        self.std_t = torch.tensor([0.229, 0.224, 0.225])
        self.t = transforms.Compose(
            [
                transforms.Resize(size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.mean_t, std=self.std_t),
            ]
        )

        if use_seg_sal:

            def seg_transform(ts: torch.Tensor):
                return ts.sum(0).clamp(0, 1)

            self.st = transforms.Compose(
                [
                    transforms.ToTensor(),
                    transforms.Lambda(seg_transform),
                ]
            )
        else:

            def seg_transform(ts: torch.Tensor):
                g = ts.sum(0)
                rs = data_utils.min_max_norm_matrix(g)
                return torch.where(rs < 0.95, 0, rs)

            self.st = transforms.Compose(
                [
                    transforms.Resize(size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.mean_t, std=self.std_t),
                    transforms.Lambda(seg_transform),
                ]
            )

        self.val_dataset = datasets.ImageFolder(self.data_path, transform=self.t)

        self.val_seg_dataset = datasets.ImageFolder(self.seg_path, transform=self.st)

        self.class_label = np.array(self.val_dataset.imgs)[:, 1].astype(np.int32)

        self.val_idx_path = os.path.join(current_path, "valid_val.npy")
        if os.path.exists(self.val_idx_path) and self.use_valid_val:
            with open(self.val_idx_path, "rb") as f:
                valid_val = np.load(f)
            self.val_dataset = torch.utils.data.Subset(self.val_dataset, valid_val)

            self.val_seg_dataset = torch.utils.data.Subset(
                self.val_seg_dataset, valid_val
            )

            self.class_label = np.array(self.val_dataset.dataset.imgs)[:, 1].astype(
                np.int32
            )[valid_val]
