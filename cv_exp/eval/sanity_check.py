import numpy as np
from cv_exp.utils import *
# import torch.nn.functional as F
from scipy import stats
from torch.utils.data import DataLoader, Subset, RandomSampler
import copy


def get_sanity_check_score(
        model: torch.nn.Module, images, targets, saliency_maps,
        func, func_args, debug=False, partitions=20):

    initial_state = copy.deepcopy(model.state_dict())

    layers = []

    try:
        layers = get_layers(model, images[:1])

        layers = layers[::-1]
        layers = np.array_split(np.array(layers), min(partitions, len(layers)))

        mprt = []
        n, w, h = saliency_maps.shape
        saliency_maps = saliency_maps.cpu().numpy()

        mprt_rs = np.empty(shape=(n, partitions))
        for i in range(partitions):
            for l in layers[i]:
                l.reset_parameters()
            rand_saliency_maps_i = func(
                model, images, targets, **func_args).cpu().numpy()

            for j in range(n):
                mprt_rs[j, i] = np.abs(stats.spearmanr(
                    saliency_maps[j].flatten(),
                    rand_saliency_maps_i[j].flatten(),
                ).statistic)
    finally:
        model.load_state_dict(initial_state)
        model.eval()

    return {
        'MPRT': mprt_rs,
    }


def batch_sanity(
    settings,
    model_key, dataset_key,
    start, end, batch_size, maps, device, func, sa_args,
    num_samples=300,
):

    model = settings.get_model(model_key).to(device).eval()
    all_idx = [i for i in range(start, end)]
    dataset = settings.get_dataset(dataset_key).val_dataset
    dataset = Subset(dataset, all_idx)
    dataset = ImageTargetSaliencyMapDataset(dataset, maps[start:end])

    all_mprt = []
    random_sampler = RandomSampler(dataset, num_samples=num_samples)
    dataloader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, **data_utils.get_data_loader_args(), sampler=random_sampler)

    num_samples = min(num_samples, int(len(dataset) * 0.1))

    pbar = tqdm(total=num_samples,
                bar_format='Model Parameter Randomization Test (MPRT): {l_bar}{bar:30}{r_bar}{bar:-10b}')

    for batch in dataloader:
        images, targets, saliency_maps = batch[0].to(
            device), batch[1].to(device), batch[2].to(device)

        mprt = get_sanity_check_score(
            model, images, targets, saliency_maps, func, sa_args)['MPRT']

        all_mprt.extend(mprt)
        pbar.update(images.shape[0])

    return {
        "mprt": np.stack(all_mprt),
    }
