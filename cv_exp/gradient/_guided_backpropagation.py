from .base import *


def guided_back_propagation(model, images, targets=None,
                            ifabs=False, aggregation='sum',
                            direction='positive',
                            iteration=1, **kwargs):
    '''
    https://github.com/vectorgrp/coderskitchen-xai/blob/main/part2_Torch_guided_backprop.ipynb
    https://www.coderskitchen.com/guided-backpropagation-with-pytorch-and-tensorflow/
    '''
    handles = []
    try:
        for i, module in enumerate(model.modules()):
            if isinstance(module, torch.nn.ReLU):
                setattr(module, 'inplace', False)
                # original guidance
                if direction == 'positive' and not ifabs:
                    # print(1)
                    handles.append(module.register_full_backward_hook(
                        lambda m, grad_in, grad_out: (
                            torch.clamp(grad_in[0], min=0.),)
                    ))
                if direction == 'negative' and not ifabs:
                    # print(2)
                    handles.append(module.register_full_backward_hook(
                        lambda m, grad_in, grad_out: (
                            torch.clamp(grad_in[0], max=0.),)
                    ))
                if direction == 'both' and not ifabs:
                    # print(3)
                    # same as vanilla graident
                    handles.append(module.register_full_backward_hook(
                        lambda m, grad_in, grad_out: (grad_in[0],)
                    ))
                if direction == 'abs' or ifabs:
                    # print(4)
                    # this will ruin the propagations
                    handles.append(module.register_full_backward_hook(
                        lambda m, grad_in, grad_out: (
                            grad_in[0].abs(),)
                    ))

        images = images.clone().detach()
        images.requires_grad_(True)

        grads_l = []
        for i in range(iteration):
            grads = get_gradients(model, images, targets, **kwargs)
            # if ifabs:
            #     grads_l.append(grads.abs())
            # else:
            grads_l.append(grads)

        grads = torch.stack(grads_l).sum(0)
        model.zero_grad()
    finally:
        for h in handles:
            h.remove()
    if aggregation == 'sum':
        return data_utils.min_max_norm_matrix(grads.abs().sum(1))
    else:
        return data_utils.min_max_norm_matrix(grads.abs().mean(1))


def misplaced_gb(model, images, targets=None,
                 ifabs=False, aggregation='mean',
                 direction='positive',
                 remainprecentage=0.05, mispercentage=0.1,
                 iteration=1, **kwargs):
    """
    Reversed GB
    """
    maps = guided_back_propagation(
        model, images, targets, ifabs=ifabs,
        aggregation=aggregation, direction=direction, iteration=iteration, **kwargs)

    return return_misplaced_map(maps, remainprecentage, mispercentage)
