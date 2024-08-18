import time
import gc
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from .utils import data_utils
from tqdm import tqdm


class Pipe:

    def __init__(self, random_seed=None) -> None:
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        if torch.backends.mps.is_built() and torch.backends.mps.is_available():
            self.device = torch.device("mps")

        if random_seed is not None:
            data_utils.fix_seed(random_seed)

        print('Using device: ', self.device)

    def get_saliency_map(
            self,
            settings,
            model_key, dataset_key,
            start, end, batch_size,
            target_exp,
            func, sa_args, use_predicted_target,
            debug=False):

        all_idx = [i for i in range(start, end)]

        model = settings.get_model(model_key).to(self.device).eval()

        dataset = settings.get_dataset(dataset_key).val_dataset
        dataset = Subset(dataset, all_idx)

        tt0 = 0
        maps = []

        pbar = tqdm(total=len(all_idx),
                    bar_format='{l_bar}{bar:30}{r_bar}{bar:-10b}')

        if 'seg_dataset' in func.__code__.co_varnames:
            dataset = data_utils.IndexedDataset(dataset)
            seg_dataset = settings.get_dataset(dataset_key).val_seg_dataset

        dataloader = DataLoader(
            dataset, batch_size=batch_size,
            shuffle=False, **data_utils.get_data_loader_args())

        indices = None
        for batch in dataloader:
            if 'seg_dataset' in func.__code__.co_varnames:
                images, targets, indices = batch[0].to(
                    self.device), batch[1].to(self.device), batch[2].to(self.device)
            else:
                images, targets = batch[0].to(
                    self.device), batch[1].to(self.device)
            if use_predicted_target:
                targets = None

            st0 = time.time()
            if target_exp == 'sa':
                if 'target_layers' in func.__code__.co_varnames:
                    saliency_map = func(
                        model, images, targets,
                        target_layers=settings.get_model_taget_layer(model_key), **sa_args)
                else:
                    if indices is not None:
                        saliency_map = func(
                            model, images, targets, indices, seg_dataset)
                    else:
                        saliency_map = func(model, images, targets, **sa_args)
                et0 = time.time() - st0
                tt0 += et0
                saliency_map = saliency_map.cpu().numpy()
                maps.extend(saliency_map)

            pbar.update(images.shape[0])

        return np.array(maps), tt0
