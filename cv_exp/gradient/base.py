import random
import torch
import time
import math
import torch.nn.functional as F
from cv_exp.utils import *
from torchvision.transforms import v2
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)


def torch_gaussian_blur(image, sigma, radius=None):
    if radius is not None:
        blurrer = v2.GaussianBlur(2 * radius + 1, sigma=sigma)
    else:
        blurrer = v2.GaussianBlur(2 * int(sigma) + 1, sigma=sigma)
    return blurrer(image)


def get_gradients(model: torch.nn.Module,
                  images: torch.Tensor,
                  targets: torch.Tensor, loss=False,
                  task='classification', pred_fn=None, outputs_agg_fn=None, with_pred=False):
    '''
    Based function for gradient calculation
    '''
    original_model_mode_on_train = model.training
    model.eval()

    images = images.clone().detach()
    images = images.requires_grad_(True)
    if task == 'detection':
        # model = model.train()
        # outputs = model(images, targets)
        pass
    else:
        outputs = model(images) if pred_fn is None else pred_fn(model, images)
    if task == 'detection':
        assert targets is not None
        assert outputs_agg_fn is not None
        # outputs = model(images, targets)
        # agg = sum(outputs.values())

        outputs = pred_fn(model, images)
        agg = outputs_agg_fn(outputs)

    elif task == 'classification' or task is None:
        if targets is None:
            # targets = (outputs.data.max(1, keepdim=True)[1]).flatten()
            targets = torch.argmax(outputs, dim=1)
        if loss:
            outputs = torch.log_softmax(outputs, 1)
            agg = F.nll_loss(outputs, targets, reduction='sum')
        else:
            agg = -1. * F.nll_loss(outputs, targets, reduction='sum')

    model.zero_grad()
    # Gradients w.r.t. input and features
    gradients = torch.autograd.grad(
        # outputs=agg, inputs=images, retain_graph=False)[0]
        outputs=agg, inputs=images, retain_graph=False,
        # outputs=outputs.max(), inputs=images, retain_graph=False,
        create_graph=False, allow_unused=True)[0]
    outputs_arr = outputs.detach().cpu()
    del agg, outputs, images
    if original_model_mode_on_train:
        model = model.train()

    if with_pred:
        return gradients, outputs_arr
    else:
        return gradients


def return_misplaced_map(maps, remainprecentage, mispercentage):
    new_maps = []
    for ii, c in enumerate(maps):
        cf = c.flatten()
        v, original_idx = torch.sort(cf)
        remain_size = int(len(v) * remainprecentage)
        mis_placed_size = int(len(v) * mispercentage)
        low_idx = np.array([i for i in range(mis_placed_size)])
        high_idx = np.array(
            [i for i in range(len(v) - 1 - remain_size - mis_placed_size, len(v) - 1 - remain_size)])
        random.shuffle(low_idx)
        random.shuffle(high_idx)

        low_original_value = v[low_idx]
        high_original_value = v[high_idx]
        cf[original_idx[low_idx]] = high_original_value
        cf[original_idx[high_idx]] = low_original_value

        new_maps.append(cf.reshape(*maps[0].shape))

    return torch.stack(new_maps)


def aggregate_saliency_maps(cams, ifabs=False, th=None, aggregation='mean'):
    """
    Aggregate the saliency maps by leveraging the magnitude of the gradient

    cams: keeps the saliency maps from multiple noise introductions
    th: is the p parameter in the paper

    """
    cams_k = torch.stack(cams, dim=4).detach()
    n, c, w, h, num_samples = cams_k.shape

    final_cam = torch.zeros(size=(n, c, w, h)).to(cams_k.device)
    # AbsoluteGrad
    if aggregation in ['mean', None]:
        for cam in cams:
            if ifabs:
                final_cam += cam.abs()
            else:
                final_cam += cam

    if aggregation == 'var':
        final_cam = cams_k.var(4)

    if aggregation == 'guided':
        if th is None:
            th = 0.7

        if th > 0:
            # equation 5 and 6
            var = cams_k.var(4)

            # this keeps the guide for differenct samples
            q = torch.quantile(var.reshape(n, c, w * h),
                               th, dim=2, keepdim=True)
            vs = torch.where(var > q.reshape(
                n, c, 1, 1).repeat(1, 1, w, h),
                1, var).reshape(n, c, w, h)
        elif th < 0:
            vs = cams_k.var(4)
        else:
            vs = torch.full(size=(n, 3, w, h),
                            fill_value=1, device=cams_k.device)
        # plotting_utils.plot_hor(vs.cpu().permute(0, 2, 3, 1))

        for i, cam in enumerate(cams):
            if ifabs:
                final_cam += (cam.abs() * vs)
            else:
                final_cam += (cam * vs)

    # line 16 of Algorithm 1
    return data_utils.min_max_norm_matrix(final_cam.abs().sum(1))


def saliency_enhance(
        model: torch.nn.Module,
        images: torch.Tensor,
        targets: torch.Tensor,
        fu=None,
        fu_args=None,
        saliency_maps: torch.Tensor = None,
        percentage=None, up_scale=[10, 20], down_scale=[0.5, 0.7]):

    if fu is not None:
        saliency_maps = fu(model, images, targets, **fu_args)
    else:
        saliency_maps = saliency_maps.clone()

    if percentage is not None:
        for i, saliency_map in enumerate(saliency_maps):
            percentile = get_quantiles(saliency_map, [percentage])[0]
            higher = torch.where(saliency_map > percentile, saliency_map, 0)
            lower = torch.where(saliency_map <= percentile, saliency_map, 0)

            r1, r2 = up_scale
            scale_map = torch.FloatTensor(
                *saliency_map.shape).uniform_(r1, r2).to(saliency_map.device)
            # up the lower part
            lower = (lower * scale_map).clamp(0, percentile)

            r1, r2 = down_scale
            scale_map = torch.FloatTensor(
                *saliency_map.shape).uniform_(r1, r2).to(saliency_map.device)
            # down the higher part
            higher = higher * scale_map
            higher = torch.where((higher >= 0) & (
                higher < percentile), percentile, higher)
            higher = torch.where(higher < 0, 0, higher)

            saliency_maps[i] = higher + lower

    return data_utils.min_max_norm_matrix(saliency_maps)


def IDGI(Gradients, Predictions):
    """
    IDGI algorithm:

    The IDGI is compatible with any IG based method, e.g., Integrated gradients (IG), Guided Integrated gradients (GIG), Blur Integrated gradients (BlurIG), ....
    For more detail, please check our paper: 
    Args:
        Gradients (list of np.array or np.array): All the gradients that are computed from the Integraded gradients path.
                                                  For instance, when compute IG, the gradients are needed for each x_j on the path. e.g. df_c(x_j)/dx_j.
                                                  Gradients is the list (or np.array) which contains all the computed gradients for the IG-base method, 
                                                  and each element in Gradients is the type of np.array.
        Predictions (list of float or np.array): List of float numbers.
                                                 Predictions contains all the predicted value for each points on the path of IG-based methods.
                                                 For instance, the value of f_c(x_j) for each x_j on the path.
                                                 Predictions is the list (or np.array) which contains all the computed target values for IG-based method, 
                                                 and each element in Predictions is a float.

    Return:
        IDGI result: Same size as the gradient, e.g., Gradients[0]
    """
    assert len(Gradients) == len(Predictions)

    idgi_result = np.zeros_like(Gradients[0])
    for i in range(len(Gradients) - 1):
        # We ignore the last gradient, e.g., the gradient of on the original image, since IDGI requires the prediction difference component, e.g., d.
        d = Predictions[i+1] - Predictions[i]
        element_product = Gradients[i]**2
        idgi_result += element_product*d/np.sum(element_product)
    return idgi_result
