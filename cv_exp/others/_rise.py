import numpy as np
from skimage.transform import resize
import torch
from torch.utils.data import DataLoader
from cv_exp.utils import *
import torch.nn.functional as F


def generate_masks(num_masks, scale_down, probability, model_input_size):
    cell_size = np.ceil(np.array(model_input_size) / scale_down)
    up_size = (scale_down + 1) * cell_size

    grid = np.random.rand(num_masks, scale_down, scale_down) < probability
    grid = grid.astype('float32')

    masks = np.empty((num_masks, * model_input_size))

    # for i in tqdm(range(num_masks), desc='Generating masks'):
    for i in range(num_masks):
        # Random shifts
        x = np.random.randint(0, cell_size[0])
        y = np.random.randint(0, cell_size[1])
        # Linear upsampling and cropping
        masks[i, :, :] = resize(grid[i], up_size, order=1, mode='reflect',
                                anti_aliasing=False)[x:x + model_input_size[0], y:y + model_input_size[1]]
    masks = masks.reshape(-1, *model_input_size, 1)
    return masks


def rise(
        model: torch.nn.Module, images: torch.Tensor, targets: torch.Tensor,
        num_masks=10, scale_down=8, probability=0.5, batch_size=64):
    model.eval()
    n, c, w, h = images.shape
    # print(images.shape)
    images = images.reshape(n, 1, c, w, h)
    masks = generate_masks(num_masks, scale_down,
                           probability, (w, h)).transpose(0, 3, 1, 2)

    masks = torch.tensor(masks, device=images.device, dtype=images.dtype)
    preds = []
    # Make sure multiplication is being done for correct axes
    # print(images.shape, masks.shape)
    masked = images * masks
    # print(masked.shape)
    # plot_hor([clp(i) for i in masked[0].cpu()])
    # for i in tqdm(range(0, num_masks, batch_size), desc='Explaining'):
    masked = masked.reshape(-1, c, w, h)
    dataloader = DataLoader(TensorDataset(masked), batch_size=batch_size,
                            shuffle=False, **data_utils.get_data_loader_args())
    with torch.no_grad():
        for batch in dataloader:
            o = model(batch)
            o = F.softmax(o, dim=1)
            preds.append(o)
    preds = torch.vstack(preds).reshape(n, num_masks, -1)
    # print(preds.shape)

    sals = []
    for i, pred in enumerate(preds):
        # (2000, 1000) (1000, 2000) (2000, 50176) (1000, 50176)
        # print(pred.shape, pred.T.shape, masks.reshape(num_masks, -1).shape)
        sal = pred.T.mm(masks.reshape(num_masks, -1)
                        ).reshape(-1, w, h)
        sal = sal / num_masks / probability
        sals.append(sal[targets[i]])
    return data_utils.min_max_norm_matrix(torch.stack(sals).to(images.device))
