from .base import *


def guided_absolute_grad(model, images, targets,
                         loss=False, num_samples=10, th=0,
                         ifabs=True,
                         ifsquare=False,
                         aggregation='guided',
                         std_spread=0.15,
                         **kwargs):
    cams = []
    std_dev = std_spread * (images.max().item() - images.min().item())

    # similar to SmoothGrad
    for i in range(num_samples):
        noise_p = torch.normal(mean=torch.zeros_like(
            images).to(images.device), std=std_dev)
        mixed = images + noise_p
        ng = get_gradients(model, mixed, targets, loss=loss, **kwargs)

        if ifsquare:
            ng *= ng
        cams.append(ng)

    # aggregate the saliency maps by leveraging the magnitude of the gradient
    return aggregate_saliency_maps(cams, ifabs, th, aggregation)


def misplaced_guided_absolute_grad(
        model, images, targets, loss=False,
        num_samples=10, std_spread=0.15,
        remainprecentage=0.05, mispercentage=0.1, **kwargs):

    maps = guided_absolute_grad(
        model, images, targets, loss=loss,
        num_samples=num_samples, std_spread=std_spread, **kwargs)

    return return_misplaced_map(maps, remainprecentage, mispercentage)
