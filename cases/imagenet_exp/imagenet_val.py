
import os
import numpy as np
import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from cv_exp import data_utils

# batch_size = 100
# s = 0
# e = 0
# t = 0.2
# valid_idx = []
# with tqdm(total=len(val_dataset)) as pbar:
#     while s < len(val_dataset):
#         e = s + batch_size if (s + batch_size) < len(val_dataset) else len(val_dataset)
#         idx = [i for i in range(s, e)]
#         images, targets = cv_exp.data_utils.get_images_targets(val_dataset, device, idx)
#         o = resnet(images)
#         # print(targets)
#         # top2 = torch.topk(o, 2, dim=1)
#         # print(top2)
#         # print(np.array([o[i][targets[i]].item() for i in range(o.shape[0])]))
#         o = F.softmax(o, dim = 1)
#         top2 = torch.topk(o, 2, dim=1)
#         top2_values = top2.values.detach().cpu().numpy()
#         top2_indices = top2.indices.detach().cpu().numpy()
#         for i in range(len(targets)):
#             if targets[i].item() in top2_indices[i]:
#                 if np.min(top2_values[i]) > t:
#                     valid_idx.append(idx[i])
#                     # print(idx[i], targets[i], top2_indices[i], top2_values[i])
#         pbar.update(e - s)
#         s = e


class ImageNetVal:

    def __init__(self,
                 data_path='./data/imagenet_val') -> None:
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

        self.classes_map = classes_map

        self.mean_t = torch.tensor([0.485, 0.456, 0.406])
        self.std_t = torch.tensor([0.229, 0.224, 0.225])
        self.t = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean_t, std=self.std_t),
        ])

        def seg_transform(ts: torch.Tensor):
            # g = torch.tensor(gaussian_filter(ts.sum(0), 8))
            g = ts.sum(0)
            rs = data_utils.min_max_norm_matrix(g)
            return torch.where(rs < 0.95, 0, rs)
            # return torch.full_like(ts, fill_value=1)
            # return torch.zeros_like(ts)

        self.st = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.mean_t, std=self.std_t),
            transforms.Lambda(seg_transform),
        ])

        self.val_idx_path = os.path.join(
            'imagenet_exp', 'imagenet_val_valid_1.npy')
        if os.path.exists(self.val_idx_path):
            with open(self.val_idx_path, 'rb') as f:
                valid_val = np.load(f)
            self.val_dataset = torch.utils.data.Subset(
                datasets.ImageFolder(data_path, transform=self.t), valid_val)

            self.class_label = np.array(self.val_dataset.dataset.imgs)[
                :, 1].astype(np.int32)[valid_val]
        else:
            self.val_dataset = datasets.ImageFolder(
                data_path, transform=self.t)

            self.class_label = np.array(self.val_dataset.imgs)[
                :, 1].astype(np.int32)
