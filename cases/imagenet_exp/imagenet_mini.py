
import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets


class ImageNetMini:

    def __init__(self, data_path='./data/imagenet-mini/val') -> None:
        classes_map = {}

        with open(os.path.join(os.path.dirname(__file__), 'LOC_synset_mapping.txt')) as f:
            lines = f.readlines()
            for line in lines:
                classes_map[len(classes_map.keys())] = {
                    'encoded_label': line[:9].strip(),
                    'label': line[9:].split(',')[0].strip(),
                }

            for k in list(classes_map.keys()):
                classes_map[classes_map[k]['encoded_label']] = {
                    'label': classes_map[k]['label'].split(',')[0].strip(),
                    'idx': k
                }

        with open(os.path.join('./data/imagenet-mini/valid_val.npy'), 'rb') as f:
            valid_val = np.load(f)

        self.classes_map = classes_map
        # print(classes_map[0])
        # print(classes_map['n01440764'])
        # print(class_label[0], classes_map[class_label[0]])

        self.mean_t = torch.tensor([0.485, 0.456, 0.406])
        self.std_t = torch.tensor([0.229, 0.224, 0.225])

        self.t = transforms.Compose([
            transforms.Resize((256, 256)),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean_t, std=self.std_t),
        ])

        self.val_dataset = torch.utils.data.Subset(
            datasets.ImageFolder(data_path, transform=self.t), valid_val)

        self.class_label = np.array(self.val_dataset.dataset.imgs)[
            :, 1].astype(np.int32)[valid_val]
