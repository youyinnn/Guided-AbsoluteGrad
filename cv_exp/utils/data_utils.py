
import time
import os
import numpy as np
import torch
from PIL import Image
import random
from torch.utils.data import Dataset
from tqdm.auto import tqdm

# ANCHOR: Data

torch.set_printoptions(precision=10, sci_mode=False)


class ImageTargetSaliencyMapDataset(Dataset):
    """
    For Batch Dataloader of RCAP
    """

    def __init__(self, dataset, saliency_maps):
        self.dataset = dataset
        self.saliency_maps = saliency_maps

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        saliency_map = self.saliency_maps[idx]
        return image, label, saliency_map


class IndexedDataset(Dataset):

    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        image, label = self.dataset[idx]
        return image, label, idx


class TensorDataset(Dataset):
    """
    For Dataloader of RCAP or ...
    """

    def __init__(self, ts):
        self.ts = ts

    def __len__(self):
        return len(self.ts)

    def __getitem__(self, idx):
        image = self.ts[idx]
        return image


def get_quantiles(saliency_map, percentage, remove_head=False):
    ury_f = saliency_map.flatten()
    v, _ = torch.sort(ury_f)
    # print(rr)
    q_idx = [int(v.shape[0] * rate) - 1 for rate in percentage]
    """
    Different partitions
    """
    q = np.array([v[i].item() for i in q_idx])
    # plt.plot(v.cpu().numpy())
    # plt.show()
    q = np.flip(q)
    # print(len(q), set(q), len(set(q)))
    if remove_head and (len(q) == len(set(q))) and (len(q) > 1):
        if q[0] == 1:
            q = q[1:]
    return q


def fix_seed(s=0):
    torch.manual_seed(s)
    torch.cuda.manual_seed_all(s)
    random.seed(s)
    np.random.seed(s)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normm_i_t(tensors):
    inputs_shape, _, w, h = tensors.shape
    mean_i_t = torch.stack([torch.full((inputs_shape,  w, h), 0.485), torch.full(
        (inputs_shape,  w, h), 0.456), torch.full((inputs_shape,  w, h), 0.406)], dim=1).to(tensors.device)
    std_i_t = torch.stack([torch.full((inputs_shape, w, h), 0.229), torch.full(
        (inputs_shape, w, h), 0.224), torch.full((inputs_shape, w, h), 0.225)], dim=1).to(tensors.device)
    return (tensors - mean_i_t) / std_i_t


def denormm_i_t(tensors):
    inputs_shape, _, w, h = tensors.shape
    mean_i_t = torch.stack([torch.full((inputs_shape,  w, h), 0.485), torch.full(
        (inputs_shape,  w, h), 0.456), torch.full((inputs_shape,  w, h), 0.406)], dim=1).to(tensors.device)
    std_i_t = torch.stack([torch.full((inputs_shape, w, h), 0.229), torch.full(
        (inputs_shape, w, h), 0.224), torch.full((inputs_shape, w, h), 0.225)], dim=1).to(tensors.device)
    return tensors * std_i_t + mean_i_t


def normm(np_arr):
    return (np_arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])


def normm_i(np_arr):
    inputs_shape = np_arr.shape[0]
    mean_i = np.stack([np.full((inputs_shape,  256, 256), 0.485), np.full(
        (inputs_shape,  256, 256), 0.456), torch.full((inputs_shape,  256, 256), 0.406)], axis=1)
    std_i = np.stack([np.full((inputs_shape, 256, 256), 0.229), np.full(
        (inputs_shape, 256, 256), 0.224), np.full((inputs_shape, 256, 256), 0.225)], axis=1)

    return (np_arr - mean_i) / std_i


def denormm(np_arr):
    return np_arr * np.array([0.229, 0.224, 0.225]) + np.array([0.485, 0.456, 0.406])


def denormm_i(np_arr):
    inputs_shape = np_arr.shape[0]
    mean_i = np.stack([
        np.full((inputs_shape,  256, 256), 0.485),
        np.full((inputs_shape,  256, 256), 0.456),
        torch.full((inputs_shape,  256, 256), 0.406)],
        axis=1)
    std_i = np.stack([np.full((inputs_shape, 256, 256), 0.229), np.full(
        (inputs_shape, 256, 256), 0.224), np.full((inputs_shape, 256, 256), 0.225)], axis=1)

    return np_arr * std_i + mean_i


def entropy(gray):
    hist, _ = np.histogram(gray, bins=256)
    hist = hist / float(gray.size)
    hist = hist[hist != 0]
    entropy_value = -np.sum(hist * np.log2(hist))
    return entropy_value


def entropy_rgb(rgb):
    return np.array([entropy(rgb[i]) for i in range(3)]).mean()


def entropy_change(images, mixed):
    imgaes_np = images.to('cpu').detach().numpy()
    mixed_np = mixed.to('cpu').detach().numpy()

    images_entropy = np.array([entropy_rgb(img) for img in imgaes_np])
    mixed_entropy = np.array([entropy_rgb(img) for img in mixed_np])
    # images_entropy = np.array([entropy(img) for img in imgaes_np])
    # mixed_entropy = np.array([entropy(img) for img in mixed_np])

    return images_entropy - mixed_entropy


mean_t = torch.tensor([0.485, 0.456, 0.406])
std_t = torch.tensor([0.229, 0.224, 0.225])


def build_input(inputs, device, norm=False):
    if norm:
        input = (torch.tensor(np.array([input.astype(
            np.float32) for input in inputs]), device=device) - mean_t.to(device)) / mean_t.to(device)
    else:
        input = torch.tensor(
            np.array([input.astype(np.float32) for input in inputs]), device=device)
    return input.transpose(2, 3).transpose(1, 2)


def read_array(instance_path, read_file_name):
    ext = '.npz' if os.path.exists(os.path.join(
        instance_path, 'maps.npz')) else '.npy'

    np_file = os.path.join(instance_path, f'{read_file_name}{ext}')
    if os.path.exists(np_file):
        with open(np_file, 'rb') as f:
            array = np.load(f, allow_pickle=True)[
                'arr_0'] if ext == '.npz' else np.load(f, allow_pickle=True)
        return array


def norm_noise(sigma, n, fold=True, channel=3):
    mu = 1  # mean
    # Generate the samples from the normal distribution
    if channel == 3:
        samples = np.random.normal(
            mu, sigma, size=(n, 3, 256, 256)).astype(np.float32)
    else:
        samples = np.random.normal(
            mu, sigma, size=(n, 256, 256)).astype(np.float32)
        samples = np.stack([samples for i in range(3)], axis=1)

    samples = np.where(samples < 0, np.abs(samples), samples)
    if not fold:
        samples = np.where(samples > 2, 2 - (samples - 2), samples)
    if fold:
        samples = np.where(samples < 0, np.abs(samples), samples)
        samples = np.where(samples > 1, np.abs(2 - samples), samples)
    return samples


def min_max_norm(u):
    if u.sum() == 0 or len(u) < 2:
        return u
    u -= u.min()
    u /= u.max()
    return u


def min_max_norm_matrix(u, axis=None):
    if type(u) is torch.Tensor:
        umin = u.min(dim=-1, keepdim=True).values.min(dim=-
                                                      2, keepdim=True).values
        u -= umin
        umax = u.max(dim=-1, keepdim=True).values.max(dim=-
                                                      2, keepdim=True).values
        u /= umax
        if torch.isnan(u).all():
            u = torch.ones_like(u)
    else:
        # narrays
        u -= u.min(axis=axis, keepdims=True)
        u /= u.max(axis=axis, keepdims=True)
        if np.isnan(u).all():
            u = np.ones_like(u)
    return torch.nan_to_num(u)


def mask(ori, sal):
    # sal = (sal + 0.1)
    if ori.shape[0] == 3:
        ori = np.transpose(ori, (1, 2, 0))
    sal = np.where(sal > 1, 1, sal)
    g = ori * np.array(Image.fromarray(sal * 255).convert('RGB'))
    return g / 255
    # return ori


def get_images_targets(dataset, device, ran_idx):
    if dataset is None:
        return None, None
    images = torch.stack([dataset[i][0].to(device) for i in ran_idx])
    if isinstance(dataset[0][1], dict):
        targets = [dataset[i][1] for i in ran_idx]
    else:
        targets = torch.tensor([dataset[i][1]
                                for i in ran_idx], device=device)

    return images, targets


def get_data_loader_args():
    if torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return {}
    else:
        return {}


def get_layers(model, dummy_input):
    handles, layers = [], []
    named_layers = dict(model.named_modules())

    def forward_hook(module, input, output):
        layers.append(module)

    for k, v in named_layers.items():
        if hasattr(v, 'reset_parameters'):
            handles.append(v.register_forward_hook(forward_hook))

    model(dummy_input)

    for h in handles:
        h.remove()

    return layers


def set_startime(key, time_map):
    if time_map.get(key) is None:
        time_map[key] = {
            'total': 0,
            's': 0,
            'count': 0,
            'average': 0,
        }
    time_map[key]['s'] = time.time()


def log_end_time(key, time_map):
    time_map[key]['total'] += (time.time() - time_map[key]['s'])
    time_map[key]['count'] += 1
    time_map[key]['average'] = time_map[key]['total'] / time_map[key]['count']


def print_time_map(time_map):
    for k, v in time_map.items():
        v.pop('s')
        v['total'] = round(v['total'], 6)
        v['average'] = round(v['average'], 6)
        print(k, ':')
        print('\ttotal', v['total'])
        print('\taverage', v['average'])
