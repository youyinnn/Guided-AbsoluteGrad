import json
import os
import gc
import torch
from torch.utils.data import DataLoader, Subset, Dataset
import gc
import numpy as np
import torch.nn.functional as F
from cv_exp.utils import *
# from . import data_utils, plotting_utils, gradient, quantus_eval_helper
# from tqdm import tqdm
# https://github.com/microsoft/vscode-jupyter/issues/8552
# ipywidgets==7.7.2

np.set_printoptions(suppress=True)


def recovering(img, saliency_map, stop, steps):
    """
    Get the multiple recovered image,
    This is referred to I_{p_k}
    """
    rs = []
    imgg = img
    local_heat_mean = []
    local_heat_sum = []
    upper_rate = 1
    imgg = torch.zeros_like(img)
    percentage = [round(rate, 2) for rate in np.arange(stop, 1, steps)]
    quantiles = get_quantiles(saliency_map, percentage, remove_head=True)
    """
    Recover based on partitions
    """
    for i in range(len(quantiles)):
        lower_rate = quantiles[i] if quantiles[i] < 1 else 0
        mc = (saliency_map > lower_rate) & (saliency_map <= upper_rate)
        local_heat_mean.append(
            saliency_map[(saliency_map > lower_rate) & (saliency_map <= 1)].mean().item())
        local_heat_sum.append(
            saliency_map[(saliency_map > lower_rate) & (saliency_map <= 1)].sum().item())
        imgg = torch.where(mc, img, imgg)
        # plot_hor([imgg.cpu().permute(1, 2, 0).detach().numpy()])
        rs.append(imgg.reshape(1, *imgg.shape))
        upper_rate = lower_rate
    return torch.vstack(rs), local_heat_mean, local_heat_sum


def get_rcap_input(model, original_images, targets,
                   saliency_maps, steps, stop, debug=False):
    """
    RCAP core code, recover and predict

    steps refer to interval parameter in the paper
    stop refer to lower_bound parameter in the paper
    """
    device = original_images.device
    n = original_images.shape[0]
    rss = []
    number_bin = None
    original_images = data_utils.denormm_i_t(original_images)
    local_heat_mean = []
    local_heat_sum = []
    recovered_imgs = []
    # one by one
    for i in range(n):
        img = original_images[i]
        saliency_map = saliency_maps[i]
        rs, lhm, lhs = recovering(img, saliency_map, steps=steps, stop=stop)
        recovered_imgs.append(rs.cpu().detach().numpy())
        local_heat_mean.append(lhm)
        local_heat_sum.append(lhs)
        number_bin = rs.shape[0] + 1
        rs = torch.vstack([rs, img.reshape(1, *img.shape)])
        if debug:
            plotting_utils.plot_hor([np.transpose(imgg.cpu().detach().numpy(), (1, 2, 0))
                                    for imgg in rs[:-1]])
        rss.append(rs)
    rss = torch.vstack(rss).to(device)
    rss = data_utils.normm_i_t(rss)
    recovered_imgs = np.array(recovered_imgs)

    if debug:
        print('Recovered Images: ', recovered_imgs.mean(), recovered_imgs.std())

    local_heat_mean = torch.tensor(local_heat_mean, device=device)
    local_heat_sum = torch.tensor(local_heat_sum, device=device)
    overall_heat_mean = saliency_maps.mean(dim=(1, 2))
    overall_heat_sum = saliency_maps.sum(dim=(1, 2))

    """
    Predict on recovered images
    """
    with torch.no_grad():

        dataloader = DataLoader(
            TensorDataset(rss), batch_size=128,
            shuffle=False, **data_utils.get_data_loader_args())

        prediction = []
        for rs in dataloader:
            p = model(rs)
            prediction.extend(p)

        prediction = torch.stack(prediction).to(device)
        # prediction= model(rss)
        # n, number_bin, n_classes;
        prediction = prediction.reshape(n, number_bin, prediction.shape[1])
        pred_score = []
        for i, pp in enumerate(prediction):
            pred_score.extend(pp[:, targets[i]])
        pred_score = torch.tensor(
            pred_score, device=device).reshape(n, number_bin)
        original_pred_score = pred_score[:, -1:]
        recovered_pred_score = pred_score[:, :-1]
        sm = F.softmax(prediction, dim=2)
        pred_prob = []
        for i, smm in enumerate(sm):
            pred_prob.extend(smm[:, targets[i]])
        pred_prob = torch.tensor(
            pred_prob, device=device).reshape(n, number_bin)
        original_pred_prob = pred_prob[:, -1:]
        recovered_pred_prob = pred_prob[:, :-1]

    """
    Get all the scores: original score, recovered scores;
    Get all the saliency mean values;
    """
    return original_pred_score.cpu().detach().numpy(), recovered_pred_score.cpu().detach().numpy(), \
        original_pred_prob.cpu().detach().numpy(), recovered_pred_prob.cpu().detach().numpy(), \
        local_heat_mean.cpu().detach().numpy(), local_heat_sum.cpu().detach().numpy(), \
        overall_heat_mean.cpu().detach().numpy(), overall_heat_sum.cpu().detach().numpy(), \
        recovered_imgs


def batch_rcap(
    settings,
    model_key, dataset_key,
    start, end, batch_size, maps, device, stop=0.7, steps=0.05
):
    """
    Get and save rcap input
    """
    model = settings.get_model(model_key).to(device).eval()
    all_idx = [i for i in range(start, end)]
    dataset = settings.get_dataset(dataset_key).val_dataset
    dataset = Subset(dataset, all_idx)
    dataset = ImageTargetSaliencyMapDataset(dataset, maps)

    # batch_size = 10
    pbar = tqdm(total=len(all_idx),
                bar_format='RCAP: {l_bar}{bar:30}{r_bar}{bar:-10b}')
    all_original_pred_score = []
    all_recovered_pred_score = []
    all_original_pred_prob = []
    all_recovered_pred_prob = []
    all_local_heat_mean = []
    all_local_heat_sum = []
    all_overall_heat_mean = []
    all_overall_heat_sum = []

    all_e = []
    dataloader = DataLoader(
        dataset, batch_size=batch_size,
        shuffle=False, **data_utils.get_data_loader_args())

    for batch in dataloader:
        images, targets, saliency_maps = batch[0].to(
            device), batch[1].to(device), batch[2].to(device)

        if settings.get_dataset('coco_imagenet') is not None:
            # coco
            targets = torch.argmax(model(images), dim=1)

        energy = np.array([data_utils.entropy(n)
                           for n in saliency_maps.cpu().detach().numpy()])

        recovered_pred = get_rcap_input(
            model, images, targets, saliency_maps, stop=stop, steps=steps)
        original_pred_score, recovered_pred_score, \
            original_pred_prob, recovered_pred_prob, \
            local_heat_mean, local_heat_sum, \
            overall_heat_mean, overall_heat_sum, \
            recovered_imgs = recovered_pred

        all_original_pred_score.extend(original_pred_score)
        all_recovered_pred_score.extend(recovered_pred_score)
        all_original_pred_prob.extend(original_pred_prob)
        all_recovered_pred_prob.extend(recovered_pred_prob)

        all_local_heat_mean.extend(local_heat_mean)
        all_local_heat_sum.extend(local_heat_sum)
        all_overall_heat_mean.extend(overall_heat_mean)
        all_overall_heat_sum.extend(overall_heat_sum)

        all_e.extend(energy)
        pbar.update(images.shape[0])

    all_original_pred_score = np.array(all_original_pred_score)
    all_recovered_pred_score = np.array(all_recovered_pred_score)
    all_original_pred_prob = np.array(all_original_pred_prob)
    all_recovered_pred_prob = np.array(all_recovered_pred_prob)

    all_local_heat_mean = np.array(all_local_heat_mean)
    all_local_heat_sum = np.array(all_local_heat_sum)
    all_overall_heat_mean = np.array(all_overall_heat_mean)
    all_overall_heat_sum = np.array(all_overall_heat_sum)

    all_e = np.array(all_e)

    return {
        'original_pred_score': all_original_pred_score,
        'recovered_pred_score': all_recovered_pred_score,
        'original_pred_prob': all_original_pred_prob,
        'recovered_pred_prob': all_recovered_pred_prob,

        'local_heat_mean': all_local_heat_mean,
        'local_heat_sum': all_local_heat_sum,
        'overall_heat_mean': all_overall_heat_mean,
        'overall_heat_sum': all_overall_heat_sum,

        "entropy": all_e,
    }


def get_rcap_score(recovered_pred, debug=False):
    """
    Calculate RCAP
    local_heat_mean refers to mean of the M_{p_k}
    recovered_pred_score refers to mean of the f(I_{p_k})
    recovered_pred_prob refers to mean of the sigma(f(I_{p_k}))
    """
    original_pred_score, recovered_pred_score, \
        original_pred_prob, recovered_pred_prob, \
        local_heat_mean, local_heat_sum, \
        overall_heat_mean, overall_heat_sum, \
        recovered_imgs = recovered_pred

    hit_rate = (
        local_heat_sum / overall_heat_sum.repeat(local_heat_mean.shape[-1]).reshape(*recovered_pred_score.shape))

    score_lhm_hr_rpp = \
        (local_heat_mean * hit_rate * recovered_pred_score).mean(-1)
    score_lhm_rpp = \
        (local_heat_mean * recovered_pred_score).mean(-1)

    prob_lhm_hr_rpp = \
        (local_heat_mean * hit_rate * recovered_pred_prob).mean(-1)
    prob_hr_rpp = \
        (hit_rate * recovered_pred_prob).mean(-1)
    prob_hr_rpp2 = \
        (hit_rate + recovered_pred_prob).mean(-1)
    # prob_lhm_rpp = \
    #     (local_heat_mean * recovered_pred_prob).mean(-1)

    if debug:
        # print('1-- local heat mean')
        # print(local_heat_mean)
        # print(local_heat_sum)

        # print('2-- overall heat sum')
        # print(overall_heat_sum)

        print('\r\n3-- hit_rate = local_heat_sum / overall_heat_sum')
        print(hit_rate)

        # print('\r\n4-- removed pred score')
        # print(recovered_pred_score)
        print('\r\n4-- removed pred prob')
        print(recovered_pred_prob, np.mean(recovered_pred_prob, axis=1))

        # print('\r\n5-- pre rs 1: local_heat_mean * hit_rate * recovered_pred_score')
        # print(local_heat_mean * hit_rate * recovered_pred_score)
        # print('\r\n5-- pre rs 2: local_heat_mean * recovered_pred_score')
        # print(local_heat_mean * recovered_pred_score)

        print('\r\n6-- rs')
        # print('local_heat_mean * hit_rate * recovered_pred_score', score_lhm_hr_rpp)
        # print('local_heat_mean * recovered_pred_score', score_lhm_rpp)
        print('local_heat_mean * hit_rate * recovered_pred_prob', prob_lhm_hr_rpp)
        print('hit_rate * recovered_pred_prob', prob_hr_rpp)
        print('hit_rate + recovered_pred_prob', prob_hr_rpp2)
        # print('local_heat_mean * recovered_pred_prob', prob_lhm_rpp)

    # eval_rs['Score: M1'] = score_lhm_hr_rpp
    # eval_rs['Score: M2'] = score_lhm_rpp
    # eval_rs['Prob: M1'] = prob_lhm_hr_rpp

    return {
        'Score: M1': score_lhm_hr_rpp,
        'Score: M2': score_lhm_rpp,
        'Prob: M1': prob_lhm_hr_rpp,
        'Prob: M3': prob_hr_rpp,
        'Prob: M4': prob_hr_rpp2,
    }
