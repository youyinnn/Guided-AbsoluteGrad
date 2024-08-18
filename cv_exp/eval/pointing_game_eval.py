from cv_exp.utils import *


def get_pg_score(saliency_maps, image_segs, q=0.7):
    count = []
    if image_segs is not None:
        for i, saliency_map in enumerate(saliency_maps):
            segmentation = image_segs[i]
            quantiles = data_utils.get_quantiles(saliency_map, [q])
            quantiles = [0] if len(quantiles) == 0 else quantiles
            idx = torch.where(saliency_map >= quantiles[0])
            centroid = (int(idx[0].float().mean()), int(idx[1].float().mean()))
            count.append(int(segmentation[centroid] == 1))

        return {
            "PG": np.array(count)
        }
    return {}
