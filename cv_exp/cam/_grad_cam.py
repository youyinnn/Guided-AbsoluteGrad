from .base import *


def grad_cam(model: torch.nn.Module, images: torch.Tensor, targets: torch.Tensor,
             target_layers, **kwargs):
    cam = None
    for c in cam_objs:
        if c['model'] == model and c['target_layers']:
            cam = c['cam']
    if cam is None:
        if images.device == torch.device('mps'):
            cam = GradCAM(
                model=model, target_layers=target_layers, use_mps=True)
        else:
            cam = GradCAM(model=model, target_layers=target_layers,
                          use_cuda=images.device == torch.device('cuda'))
        cam_objs.append({
            'model': model,
            'target_layers': target_layers,
            'cam': cam
        })

    targets = [ClassifierOutputTarget(i.item()) for i in targets]
    # st = time.time()
    grayscale_cam = cam(input_tensor=images, targets=targets, **kwargs)
    # et = time.time() - st
    # print(et)
    n, w, h = grayscale_cam.shape
    # grayscale_cam = grayscale_cam.reshape(n, 1, w, h)
    # print(grayscale_cam.shape)
    return data_utils.min_max_norm_matrix(torch.tensor(grayscale_cam, device=images.device, dtype=images.dtype))
