import numpy as np
import torch.nn.functional as F

beta = 0.25
alpha = 0.25
gamma = 2
epsilon = 1e-5
smooth = 1


class Semantic_loss_functions(object):
    def __init__(self):
        # print("semantic loss functions initialized")
        pass

    def generalized_dice_coefficient(self, y_true, y_pred):
        smooth = 1.
        y_true_f = np.array(y_true).flatten()
        y_pred_f = np.array(y_pred).flatten()
        intersection = np.sum(y_true_f * y_pred_f)
        score = (2. * intersection + smooth) / (
            np.sum(y_true_f) + np.sum(y_pred_f) + smooth)
        return score

    def dice_loss(self, y_true, y_pred):
        loss = 1 - self.generalized_dice_coefficient(y_true, y_pred)
        return loss

    def log_cosh_dice_loss(self, y_true, y_pred):
        x = self.dice_loss(y_true, y_pred)
        return np.log((np.exp(x) + np.exp(-x)) / 2.0)


slf = Semantic_loss_functions()


def get_loss(saliency_maps, image_segs):
    """
    Calculate Log-Cosh Dice Loss
    """
    # MSE Loss: per img
    loss = []
    loss_2 = []
    loss_3 = []
    if image_segs is not None:
        for i, u in enumerate(saliency_maps):
            # seg = u
            seg = image_segs[i]
            l = F.mse_loss(u, seg, reduction='mean').sqrt().item()
            loss.append(l)

            seg_np = seg.detach().cpu().numpy()
            u_np = u.detach().cpu().numpy()

            l_2 = slf.log_cosh_dice_loss(seg_np, u_np)
            loss_2.append(l_2)

            l_3 = F.l1_loss(u, seg, reduction='mean').sqrt().item()
            loss_3.append(l_3)

        return {
            'MSE': np.array(loss),
            'MAE': np.array(loss_3),
            'Log-Cosh Dice Loss': np.array(loss_2),
        }

    return {}
