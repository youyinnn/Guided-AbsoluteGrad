from .base import *


def smooth_grad(model, images, targets, loss=False,
                num_samples=10, std_spread=0.15, aggregate='mean',
                ifabs=False, ifsquare=False, **kwargs):
    """
    https://github.com/idiap/fullgrad-saliency/blob/master/saliency/smoothgrad.py
    """
    std_dev = std_spread * (images.max().item() - images.min().item())

    cams = []

    # cam = torch.zeros_like(images).to(images.device)
    for i in range(num_samples):
        noise_p = torch.normal(mean=torch.zeros_like(
            images).to(images.device), std=std_dev)
        mixed = images + noise_p
        # plotting_utils.plot_hor([plotting_utils.clp(iii)
        #                         for iii in mixed.cpu().numpy()])
        ng = get_gradients(model, mixed, targets, loss=loss, **kwargs)
        # plotting_utils.plot_hor([iii
        #                         for iii in ng.abs().sum(1).cpu().numpy()])
        if ifabs:
            ng = ng.abs()

        if ifsquare:
            ng *= ng

        # cam += (ng) / num_samples
        cams.append(ng)

    cams = torch.stack(cams, dim=4)

    if aggregate == 'mean':
        cams = cams.mean(dim=4)
    elif aggregate == 'var':
        cams = cams.var(dim=4)
    return data_utils.min_max_norm_matrix(cams.abs().sum(1))


def var_grad(model, images, targets, loss=False,
             num_samples=10, std_spread=0.15,
             ifabs=False, ifsquare=False, **kwargs):
    return smooth_grad(
        model, images, targets, loss=loss,
        num_samples=num_samples, std_spread=std_spread, ifabs=ifabs,
        ifsquare=ifsquare, aggregate='var', **kwargs)


def misplaced_smooth_grad(model, images, targets, loss=False,
                          num_samples=10, std_spread=0.15,
                          existing_map=None,
                          remainprecentage=0.05, mispercentage=0.1, **kwargs):
    """
    Reversed SG
    """
    if existing_map is None:
        maps = smooth_grad(
            model, images, targets, loss=loss,
            num_samples=num_samples, std_spread=std_spread, **kwargs)
    else:
        maps = existing_map

    return return_misplaced_map(maps, remainprecentage, mispercentage)


def one_direction_smooth_grad(model, images, targets, loss=False,
                              num_samples=10, std_spread=0.15, positive_only=True, **kwargs):
    # TODO: Merge with smoothGrad
    '''
    SmoothGrad+ and SmoothGrad- implementation
    '''
    std_dev = std_spread * (images.max().item() - images.min().item())

    cam = torch.zeros_like(images).to(images.device)
    for i in range(num_samples):
        noise_p = torch.normal(mean=torch.zeros_like(
            images).to(images.device), std=std_dev)
        mixed = images + noise_p
        ng = get_gradients(model, mixed, targets, loss=loss, **kwargs)

        if positive_only:
            ng = torch.where(ng >= 0, ng, 0)
        else:
            ng = torch.where(ng < 0, ng, 0)

        cam += (ng) / num_samples
    return data_utils.min_max_norm_matrix(cam.abs().sum(1))
