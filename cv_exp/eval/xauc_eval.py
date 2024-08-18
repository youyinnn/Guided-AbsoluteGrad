from torchvision.transforms import v2
import numpy as np
from cv_exp.utils import *
from torch.utils.data import DataLoader, Subset, Dataset
import torch.nn.functional as F
from sklearn import metrics


def __get_pred(model, dataloader, device, targets, n, num_aug):
    predition = []
    for d in dataloader:
        d_out = model(d.to(device))
        predition.extend(d_out)

    predition = torch.stack(predition).to(device)
    num_classes = predition.shape[1]
    predition = predition.reshape(n, num_aug, num_classes)
    sm = F.softmax(predition, dim=2)
    pred_prob = []
    for i, smm in enumerate(sm):
        pred_prob.extend(smm[:, targets[i]])
    pred_prob = torch.tensor(pred_prob, device=device).reshape(n, num_aug)

    return pred_prob


delete_percentage = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
# delete_percentage = [1, 0.8, 0.6, 0.4, 0.2]
# delete_percentage = np.arange(1, 0, -0.05)
insert_percentage = np.flip(delete_percentage)


def get_auc_input(
    model, original_images, targets, saliency_maps, sigma=None, debug=False
):
    n, _, w, h = original_images.shape
    device = original_images.device
    num_aug = len(delete_percentage)

    all_deleted = []
    all_inserted = []
    sigma = sigma or 16
    for i in range(len(original_images)):
        deleted = []
        inserted = []

        image, target, saliency_map = original_images[i], targets[i], saliency_maps[i]
        delete_quantiles = get_quantiles(saliency_map, delete_percentage)
        insert_quantiles = get_quantiles(
            saliency_map, insert_percentage, remove_head=False
        )

        for q in np.flip(delete_quantiles):
            d = torch.where(saliency_map > q, 0, image)
            deleted.append(d)

        blurrer = v2.GaussianBlur(kernel_size=sigma * 2 + 1, sigma=sigma)
        blurred = blurrer(image)
        # for q in np.flip(insert_quantiles):
        #     i = torch.where(saliency_map < q, image, blurred)
        for q in insert_quantiles:
            i = torch.where(saliency_map > q, image, blurred)
            inserted.append(i)

        if debug:
            plot_hor([clp(k.cpu()) for k in deleted])
            plot_hor([clp(k.cpu()) for k in inserted])

        all_deleted.extend(deleted)
        all_inserted.extend(inserted)

    all_deleted = torch.stack(all_deleted)
    all_inserted = torch.stack(all_inserted)
    with torch.no_grad():
        deleted_dataloader = DataLoader(
            TensorDataset(all_deleted),
            batch_size=4,
            shuffle=False,
            **data_utils.get_data_loader_args(),
        )

        d_pred_prob = __get_pred(model, deleted_dataloader, device, targets, n, num_aug)

        inserted_dataloader = DataLoader(
            TensorDataset(all_inserted),
            batch_size=4,
            shuffle=False,
            **data_utils.get_data_loader_args(),
        )

        i_pred_prob = __get_pred(
            model, inserted_dataloader, device, targets, n, num_aug
        )

    if debug:
        print(f"D Prob: {d_pred_prob.cpu().numpy()}")
        print(f"I Prob: {i_pred_prob.cpu().numpy()}")

    return d_pred_prob.cpu(), i_pred_prob.cpu()


def get_auc_score(d_pred_prob, i_pred_prob):
    # d_pred_prob.mean(0), i_pred_prob.mean(0)
    # print(d_pred_prob)
    # print(i_pred_prob)
    dauc = metrics.auc(insert_percentage, d_pred_prob.mean(0))
    iauc = metrics.auc(insert_percentage, i_pred_prob.mean(0))
    return {
        "DAUC": dauc,
        "IAUC": iauc,
        "Overall_AUC": iauc - dauc,
        "AUC_Percentage": insert_percentage,
        "DAUC_arr": d_pred_prob,
        "IAUC_arr": i_pred_prob,
    }


def batch_auc(settings, model_key, dataset_key, start, end, batch_size, maps, device):
    """
    Get and save rcap input
    """
    model = settings.get_model(model_key).to(device).eval()
    all_idx = [i for i in range(start, end)]
    dataset = settings.get_dataset(dataset_key).val_dataset
    dataset = Subset(dataset, all_idx)
    dataset = ImageTargetSaliencyMapDataset(dataset, maps)

    # batch_size = 10
    pbar = tqdm(total=len(all_idx), bar_format="AUC: {l_bar}{bar:30}{r_bar}{bar:-10b}")

    all_dauc_prob = []
    all_iauc_prob = []

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        **data_utils.get_data_loader_args(),
    )

    for batch in dataloader:
        images, targets, saliency_maps = (
            batch[0].to(device),
            batch[1].to(device),
            batch[2].to(device),
        )

        d_pred_prob, i_pred_prob = get_auc_input(model, images, targets, saliency_maps)

        all_dauc_prob.extend(d_pred_prob)
        all_iauc_prob.extend(i_pred_prob)
        pbar.update(images.shape[0])

    return {
        "dauc_prob": np.stack(all_dauc_prob),
        "iauc_prob": np.stack(all_iauc_prob),
    }
