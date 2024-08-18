from .base import *


def l1_distance(x1, x2):
    return torch.abs(x1 - x2).sum()


def translate_x_to_alpha(x, x_input, x_baseline):
    return torch.where(x_input - x_baseline != 0,
                       (x - x_baseline) / (x_input - x_baseline), torch.nan)


def translate_alpha_to_x(alpha, x_input, x_baseline):
    assert 0 <= alpha <= 1.0
    return x_baseline + (x_input - x_baseline) * alpha


def guided_ig_impl(model, images: torch.Tensor, targets, x_baseline=None,
                   loss=False, steps=10, fraction=0.25, ifabs=False,
                   direction='both', max_dist=0.02,
                   debug=False, plot=False, **kwargs):
    '''
    https://github.com/PAIR-code/saliency/blob/master/saliency/core/guided_ig.py
    '''

    # A very small number for comparing floating point values.
    EPSILON = 1E-9
    if x_baseline is None:
        x_baseline = torch.zeros_like(images)
    x = x_baseline.clone()
    l1_total = l1_distance(images, x_baseline)
    # attr = torch.zeros_like(images)
    attr = []

    total_diff = images - x_baseline
    if torch.abs(total_diff).sum() == 0:
        return torch.zeros_like(images)
    ss = [0, 0, 0, 0, 0]
    # Iterate through every step.
    for step in range(steps):
        st0 = time.time()
        # Calculate gradients and make a copy.
        grad_actual = get_gradients(
            model, x[None, :, :, :], targets,
            loss=loss, **kwargs)[0]

        ss[0] += time.time() - st0

        st1 = time.time()
        grad = grad_actual.clone()
        alpha = (step + 1.0) / steps
        alpha_min = max(alpha - max_dist, 0.0)
        alpha_max = min(alpha + max_dist, 1.0)
        x_min = translate_alpha_to_x(alpha_min, images, x_baseline)
        x_max = translate_alpha_to_x(alpha_max, images, x_baseline)
        l1_target = l1_total * (1 - (step + 1) / steps)
        ss[1] += time.time() - st1

        if debug and plot:
            print('===')
        gamma = torch.inf

        st2 = time.time()
        while gamma > 1.0:
            x_old = x.clone()
            x_alpha = translate_x_to_alpha(x, images, x_baseline)
            x_alpha[torch.isnan(x_alpha)] = alpha_max
            x[x_alpha < alpha_min] = x_min[x_alpha < alpha_min]

            l1_current = l1_distance(x, images)
            if math.isclose(l1_target, l1_current, rel_tol=EPSILON, abs_tol=EPSILON):
                attr += (x - x_old) * grad_actual
                break
            grad[x == x_max] = torch.inf

            threshold = torch.quantile(
                torch.abs(grad), fraction, interpolation='lower')
            s = torch.logical_and(
                torch.abs(grad) <= threshold, grad != torch.inf)

            l1_s = (torch.abs(x - x_max) * s).sum()

            if l1_s > 0:
                gamma = ((l1_current - l1_target) / l1_s)
            else:
                break
                # gramma = torch.inf

            if gamma > 1.0:
                x[s] = x_max[s]
            else:
                assert gamma > 0, gamma
                x[s] = translate_alpha_to_x(gamma, x_max, x)[s]

            rs = (x - x_old) * grad_actual

            if ifabs or direction == 'abs':
                rs = rs.abs()
            elif direction == 'both':
                pass
            elif direction == 'positive':
                rs = torch.clamp(rs, min=0)
            elif direction == 'negative':
                rs = torch.clamp(rs, max=0)

            # attr += rs
            attr.append(rs)

            # debug
            if debug and plot:
                plotting_utils.plot_hor([
                    plotting_utils.clp(x.cpu()),
                    grad_actual.abs().sum(0).cpu(),
                    s.type(torch.uint8).sum(0).cpu(),
                    (1 - s.type(torch.uint8)).sum(0).cpu(),
                    plotting_utils.clp((x - x_old).cpu()),
                    ((x - x_old) * grad_actual).abs().sum(0).cpu(),
                    attr.abs().sum(0).cpu(),
                ], subplot_titles=['x', 'grad_actual(x)', 'selected', 're-selected', 'x - x_old', '(x - x_old) * grad_actual(x)', 'attr'])

        ss[2] += time.time() - st2

    if debug:
        print(ss)
    return attr


def guided_ig(
        model, images, targets,
        loss=False,
        num_samples=5,
        direction='both',
        fraction=0.25, max_dist=0.1,
        debug=False,
        ifabs=False,
        aggregation='mean', th=0,
        **kwargs):

    if targets is None:
        targets = torch.argmax(model(images), dim=1)

    us = []
    for i, image in enumerate(images):
        # can't do batch, do it one by one
        u = guided_ig_impl(
            model, image, targets[None, i], None, loss=loss,
            steps=num_samples, fraction=fraction,
            direction=direction,
            max_dist=max_dist, debug=debug, ifabs=ifabs, **kwargs)

        us.append([uu[None, :, :, :] for uu in u])

    uss = [aggregate_saliency_maps(
        uu, ifabs=ifabs, th=th, aggregation=aggregation) for uu in us]
    final = torch.stack(uss)
    n, _, w, h = final.shape
    return final.reshape(n, w, h)
    # return data_utils.min_max_norm_matrix(us.abs().sum(1))


def misplaced_guided_ig(
    model, images, targets,
    loss=False,
    num_samples=5,
    direction='both',
    fraction=0.25, max_dist=0.1,
    debug=False,
    remainprecentage=0.05, mispercentage=0.1,
    **kwargs
):
    """
    Reversed Guided IG
    """
    maps = guided_ig(
        model, images, targets,
        loss=loss, num_samples=num_samples, direction=direction,
        fraction=fraction, max_dist=max_dist, debug=debug,
        **kwargs)

    return return_misplaced_map(maps, remainprecentage, mispercentage)
