import torch
from cv_exp.utils import *


def segmentation_exp(
    model: torch.nn.Module,
    images: torch.Tensor,
    targets: torch.Tensor,
    ids,
    seg_dataset,
):
    rs = [seg_dataset[i][0] for i in ids]
    # print(torch.stack(rs).shape)
    return torch.stack(rs).to(images.device)
