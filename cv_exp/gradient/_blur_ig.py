from .base import *


def blur_integrated_gradients(
        model: torch.nn.Module, images: torch.Tensor, targets: torch.Tensor,
        num_samples=30, radius=20,
        loss=False, max_sigma=50, grad_step=0.01,
        aggregation=None, th=0,
        ifabs=False, direction=None, ifsqrt=False, **kwargs):
    """
    https://github.com/PAIR-code/saliency/blob/master/saliency/core/blur_ig.py
    """
    if ifsqrt:
        sigmas = [math.sqrt(float(i)*max_sigma/float(num_samples)
                            ) for i in range(0, num_samples+1)]
    else:
        sigmas = [float(i)*max_sigma/float(num_samples)
                  for i in range(0, num_samples+1)]
    step_vector_diff = torch.tensor(
        [sigmas[i+1] - sigmas[i] for i in range(0, num_samples)], device=images.device)

    total_gradients = []

    et1 = 0
    et2 = 0
    for i in range(num_samples):
        st1 = time.time()
        if sigmas[i] == 0:
            x_step = images.detach().clone()
        else:
            x_step = torch_gaussian_blur(images, sigmas[i], radius)

        # plot_hor([clp(kk) for kk in x_step.cpu()])

        gaussian_gradient = (torch_gaussian_blur(images, sigmas[i] + grad_step, radius)
                             - x_step) / grad_step

        et1 += (time.time() - st1)

        st2 = time.time()
        ng = get_gradients(model, torch.tensor(
            x_step, device=images.device), targets, loss=loss, **kwargs)

        if ifabs:
            ng = ng.abs()
        elif direction in ['both', None]:
            pass
        elif direction == 'positive':
            ng = torch.clamp(ng, min=0)
        elif direction == 'negative':
            ng = torch.clamp(ng, max=0).abs()

        tmp = step_vector_diff[i] * (gaussian_gradient * ng)

        total_gradients.append(tmp)
        et2 += (time.time() - st2)

    return aggregate_saliency_maps(total_gradients, ifabs, th, aggregation)


def misplaced_blurIg(
    model: torch.nn.Module, images: torch.Tensor, targets: torch.Tensor,
        num_samples=30, radius=20,
        loss=False, max_sigma=50, grad_step=0.01,
        aggregation=None, th=0,
        ifabs=False, direction=None, ifsqrt=False,
        remainprecentage=0.05, mispercentage=0.1,
        **kwargs):
    """
    Reversed BlurIG
    """

    maps = blur_integrated_gradients(
        model, images, targets, ifabs=ifabs,
        aggregation=aggregation, direction=direction, num_samples=num_samples,
        radius=radius, loss=loss, max_sigma=max_sigma,
        grad_step=grad_step, th=th, ifsqrt=ifsqrt, **kwargs)

    return return_misplaced_map(maps, remainprecentage, mispercentage)
