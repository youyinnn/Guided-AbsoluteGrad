from .base import *
import torch.nn.functional as F


def integrated_gradients_impl(model, images, targets,
                              baseline, steps=50, batch=10,
                              direction='both',
                              ifabs=False, **kwargs):
    """
    https://github.com/ankurtaly/Integrated-Gradients/blob/master/IntegratedGradients/integrated_gradients.py
    """
    if baseline is None:
        baseline = torch.zeros_like(images)

    n, _, w, h = images.shape
    true_steps = steps + 1
    scaled_inputs = torch.vstack([baseline + (float(i) / steps) * (images-baseline)
                                  for i in range(0, true_steps)]).to(images.device)

    scaled_inputs = scaled_inputs.reshape(true_steps, n, 3, w, h)
    scaled_inputs = torch.transpose(scaled_inputs, dim0=1, dim1=0)
    all_mean_gradients = []
    targets = targets.repeat(true_steps, 1).T.flatten()
    scaled_inputs = scaled_inputs.reshape(n * true_steps, _, w, h)

    n_scaled_inputs = scaled_inputs.shape[0]
    s = 0
    gradient_list = []
    while s < n_scaled_inputs:
        e = s + batch if s + batch < n_scaled_inputs else n_scaled_inputs
        gradients = get_gradients(model, scaled_inputs[s:e],
                                  targets[s:e], loss=True, **kwargs)
        gradient_list.extend(gradients)
        s += batch
    gradients = torch.stack(gradient_list)
    # gradients.shape = (n, steps, channel, w, h)
    gradients = gradients.reshape(n, true_steps, _, w, h)

    for gradients_one in gradients:
        # gradients_one.shape = (steps, channel, w, h)
        if ifabs or direction == 'abs':
            gradients_one = gradients_one.abs()
        elif direction == 'both':
            pass
        elif direction == 'positive':
            gradients_one = torch.clamp(gradients_one, min=0)
        elif direction == 'negative':
            gradients_one = torch.clamp(gradients_one, max=0)

        # gradients_one.mean(dim=0, keepdim=True).shape
        #                                           = (1, channel, w, h)
        all_mean_gradients.append(gradients_one.mean(dim=0, keepdim=True))

    # all_mean_gradients.shape = (n, channel, w, h)
    all_mean_gradients = torch.vstack(all_mean_gradients)

    return ((images - baseline) * all_mean_gradients)


def integrated_gradients_impl_idgi(
        model, images, targets,
        baseline, steps=50, batch=10,
        **kwargs):
    """
    https://github.com/ankurtaly/Integrated-Gradients/blob/master/IntegratedGradients/integrated_gradients.py
    """
    if baseline is None:
        baseline = torch.zeros_like(images)

    n, _, w, h = images.shape
    true_steps = steps + 1
    scaled_inputs = torch.vstack([baseline + (float(i) / steps) * (images-baseline)
                                  for i in range(0, true_steps)]).to(images.device)

    scaled_inputs = scaled_inputs.reshape(true_steps, n, 3, w, h)
    scaled_inputs = torch.transpose(scaled_inputs, dim0=1, dim1=0)
    all_mean_gradients = []
    targets = targets.repeat(true_steps, 1).T.flatten()
    scaled_inputs = scaled_inputs.reshape(n * true_steps, _, w, h)

    n_scaled_inputs = scaled_inputs.shape[0]
    s = 0
    gradient_list = []
    outputs_list = []
    while s < n_scaled_inputs:
        e = s + batch if s + batch < n_scaled_inputs else n_scaled_inputs
        gradients, outputs = get_gradients(model, scaled_inputs[s:e],
                                           targets[s:e], loss=True, with_pred=True, **kwargs)
        outputs = F.softmax(outputs, dim=1)
        for i, t in enumerate(targets[s:e].cpu().numpy()):
            outputs_list.append(outputs[i][t].item())
        # plot_hor([clp(kkk) for kkk in scaled_inputs[s:e].cpu()])
        # plot_hor([gl.cpu().sum(0).abs() for gl in gradients])
        gradient_list.extend(gradients)
        s += batch

    gradients = torch.stack(gradient_list)
    # gradients.shape = (n, steps, channel, w, h)
    gradients = gradients.reshape(n, true_steps, _, w, h)

    for gradients_one in gradients:
        all_mean_gradients.append(gradients_one.mean(dim=0, keepdim=True))

    # all_mean_gradients.shape = (n, channel, w, h)
    all_mean_gradients = torch.vstack(all_mean_gradients)

    # for gg in gradients:
    #     print(gg.shape)
    #     plot_hor([gl.cpu().sum(0).abs() for gl in gg])
    return gradients, outputs_list


def integrated_gradients(model, images, targets,
                         direction='both',
                         aggregation=None,
                         th=0, idgi=False,
                         steps=5, trials=2, ifabs=False):

    all_intgrads = []
    all_gradient_list = []
    all_outputs_list = []
    model.eval()
    d = torch.distributions.uniform.Uniform(
        images.min().item(), images.max().item())
    baselines = d.sample(sample_shape=(
        trials, *images.shape)).to(images.device)

    if targets is None:
        targets = torch.argmax(model(images), dim=1)

    for i in range(trials):
        if idgi:
            gradient_list, outputs_list = integrated_gradients_impl_idgi(
                model, images, targets,
                baseline=baselines[i],
                steps=steps - 1,
            )
            all_gradient_list.extend(gradient_list)
            all_outputs_list.extend(outputs_list)
        else:
            intgrads = integrated_gradients_impl(
                model, images, targets,
                direction=direction,
                baseline=baselines[i],
                steps=steps - 1,
                ifabs=ifabs
            )
            all_intgrads.append(intgrads)

    if idgi:
        n, c, w, h = images.shape
        all_gradient_list = torch.stack(
            all_gradient_list)
        all_outputs_list = np.array(all_outputs_list).reshape(n, -1)

        all_idgi_result = []
        for i in range(n):
            g = all_gradient_list[i]
            o = all_outputs_list[i]
            idgi_result = torch.zeros_like(g[0])
            for j in range(len(g) - 1):
                d = o[j + 1] - o[j]
                element_product = g[j]**2
                # element_product = g[j]
                idgi_result += element_product * d / \
                    np.sum(element_product.cpu().numpy())
                # idgi_result += element_product

            all_idgi_result.append(idgi_result)
        all_idgi_result = torch.stack(all_idgi_result)
        return data_utils.min_max_norm_matrix(all_idgi_result.abs().sum(1))
    else:
        return aggregate_saliency_maps(all_intgrads, ifabs, th, aggregation)


def misplaced_ig(
    model, images, targets,
    direction='both',
    aggregation=None,
    th=0,
    steps=10, trials=3, ifabs=False,
    remainprecentage=0.05, mispercentage=0.1,
    **kwargs
):
    """
    Reversed IG
    """

    maps = integrated_gradients(
        model, images, targets,
        direction=direction, aggregation=aggregation, th=th,
        steps=steps, trials=trials, ifabs=ifabs,
        **kwargs)

    return return_misplaced_map(maps, remainprecentage, mispercentage)
