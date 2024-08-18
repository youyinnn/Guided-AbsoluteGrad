from .base import *


def vanilla_gradient(model, images, targets, loss=False, **kwargs):
    input_grad = get_gradients(model, images, targets, loss=loss, **kwargs)
    # input_grad = torch.minimum(input_grad, torch.zeros_like(input_grad))
    return data_utils.min_max_norm_matrix(input_grad.abs().sum(1))
