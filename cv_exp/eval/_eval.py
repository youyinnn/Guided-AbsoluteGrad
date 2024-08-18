import json
import os
import gc
import torch
import time
from cv_exp.utils import *
from .rcap_eval import *
from .xauc_eval import *
from .sanity_check import *
from .pointing_game_eval import *
from .segmentation_loss import *
import traceback


def eval_for_demo(
    model,
    images,
    image_segs,
    targets,
    stop,
    steps,
    func,
    func_args,
    ifcam=False,
    ifgau=False,
    if_sanity_check=False,
    if_auc_check=True,
    debug=False,
):

    st = time.time()
    saliency_maps = func(model, images, targets, **func_args)
    end_time = time.time() - st

    entropys = np.array(
        [data_utils.entropy(n) for n in saliency_maps.cpu().detach().numpy()]
    )

    recovered_pred = get_rcap_input(
        model, images, targets, saliency_maps, stop=stop, steps=steps, debug=debug
    )

    eval_rs = {}
    eval_rs.update(get_rcap_score(recovered_pred, debug=debug))
    eval_rs.update(
        get_loss(
            saliency_maps,
            image_segs,
        )
    )
    if if_auc_check:
        d_pred_prob, i_pred_prob = get_auc_input(
            model, images, targets, saliency_maps, debug=debug
        )
        eval_rs.update(get_auc_score(d_pred_prob, i_pred_prob))
    eval_rs.update(get_pg_score(saliency_maps, image_segs))

    if if_sanity_check:
        eval_rs.update(
            get_sanity_check_score(
                model, images, targets, saliency_maps, func, func_args, debug=debug
            )
        )

    recovered_imgs = recovered_pred[-1]
    return saliency_maps, end_time, entropys, eval_rs, recovered_imgs


def evaluation_demo(
    model,
    val_dataset,
    val_seg_dataset,
    device,
    settings,
    ran_idx,
    classes_map,
    class_label,
    stop=0.5,
    steps=0.05,
    ifquantus=False,
    quantus_args={},
    debug=False,
    random_seed=None,
    if_sanity_check=False,
    if_auc_check=False,
    use_predicted_targets=False,
):
    if random_seed is not None:
        data_utils.fix_seed(random_seed)
    images, targets = data_utils.get_images_targets(val_dataset, device, ran_idx)
    image_segs, _ = data_utils.get_images_targets(val_seg_dataset, device, ran_idx)

    saliency_maps_rs = []
    recoered_imgs_rs = []
    entropy_rs = []
    rs = []

    if use_predicted_targets:
        o = model(images)
        targets = torch.argmax(o, dim=1)

    eval_rs_map = {}
    sanity_check_rs_map = {}
    for k, setting in tqdm(settings.items(), desc="Proposed Evaluation: ", position=0):
        saliency_maps, end_time, entropys, eval_rs, recovered_imgs = eval_for_demo(
            model,
            images,
            image_segs,
            targets,
            stop,
            steps,
            *setting,
            if_auc_check=if_auc_check,
            if_sanity_check=if_sanity_check,
            debug=debug,
        )

        if eval_rs.get("MPRT") is not None:
            sanity_check_rs_map[k] = (
                None,
                k,
                None,
                None,
                dict(mprt=eval_rs.get("MPRT")),
                None,
                None,
                None,
            )
            eval_rs.pop("MPRT")

        eval_rs_map[k] = eval_rs
        saliency_maps_rs.append(saliency_maps)
        recoered_imgs_rs.append(recovered_imgs)
        rs.append([k, saliency_maps, end_time, entropys])
        entropy_rs.append([k, *entropys, np.mean(entropys)])

    aggregated_eval_rs_map = aggregate_different_eval_rs(eval_rs_map)
    aggregated_quantus_rs_map = None

    if bool(sanity_check_rs_map):
        plotting_utils.plot_sanity_check(sanity_check_rs_map)

    images_array = [plotting_utils.clp(img) for img in plotting_utils.ttonp(images)]
    plotting_utils.transpose_sa(
        len(images_array),
        [images_array, *[plotting_utils.ttonp(rss[1]) for rss in rs]],
        sa_names=["", *[rss[0] for rss in rs]],
        idx=[f"{i} {classes_map[i.item()]['label']}" for i in targets],
    )

    saliency_maps_rs_map = {}
    for i, (k, setting) in enumerate(settings.items()):
        saliency_maps_rs_map[k] = saliency_maps_rs[i]

    if image_segs is None:
        recoered_imgs_rs = np.array(recoered_imgs_rs)

    return (
        images,
        image_segs,
        saliency_maps_rs_map,
        recoered_imgs_rs,
        aggregated_eval_rs_map,
        aggregated_quantus_rs_map,
    )


def aggregate_different_eval_rs(eval_rs_map, extra_info=None):
    mmms = {
        "Mean": lambda d: np.array(d).mean(),
        "Sum": lambda d: np.array(d).sum(),
        "Original": lambda d: d,
        "Count": lambda d: np.array(d).sum() / len(d),
    }
    aggregated_eval_rs_map = {
        # 'MSE': [
        #     "Low MSE",
        #     mmms, [], "Mean", True
        # ],
        "MAE": ["MAE", mmms, [], "Mean", True],
        "Log-Cosh Dice Loss": ["Log-Cosh Dice Loss", mmms, [], "Mean", True],
        # 'Score: M1': [
        #     "Score: M1",
        #     mmms, [], "Mean", False
        # ],
        # 'Score: M2': [
        #     "Score: M2",
        #     mmms, [], "Mean", False
        # ],
        # 'Prob: M1': [
        #     "RCAP",
        #     mmms, [], "Mean", False
        # ],
        "Prob: M3": ["RCAP2", mmms, [], "Mean", False],
        # 'Prob: M4': [
        #     "RCAP3",
        #     mmms, [], "Mean", False
        # ],
        "DAUC": ["DAUC", mmms, [], "Original", False],
        "IAUC": ["IAUC", mmms, [], "Original", False],
        # 'Overall_AUC': [
        #     "Overall_AUC",
        #     mmms, [], "Original", False
        # ],
        # 'PG': [
        #     "PG",
        #     mmms, [], "Count", False
        # ],
        "MPRT": ["MPRT", mmms, [], "Mean", False],
    }

    for k, eval_rs in eval_rs_map.items():
        for eval_metric_set, data in eval_rs.items():
            if aggregated_eval_rs_map.get(eval_metric_set) is not None:
                m_and_func = aggregated_eval_rs_map[eval_metric_set][1]
                m_and_rs = aggregated_eval_rs_map[eval_metric_set][2]
                rs__ = [k]
                eval_metric_func = m_and_func[
                    aggregated_eval_rs_map[eval_metric_set][3]
                ]
                rs__.append(eval_metric_func(data))
                if extra_info is not None:
                    rs__ = [*rs__, *extra_info["data"][k]]
                m_and_rs.append(rs__)

    removed = []
    for k, v in aggregated_eval_rs_map.items():
        greater_better = not v[4]
        eval_rs_value = np.array([vvv[1] for vvv in v[2]])

        if (eval_rs_value == 0.0).all():
            gen_eval_rs_value = eval_rs_value
            removed.append(k)
        else:
            gen_eval_rs_value = data_utils.min_max_norm(eval_rs_value)

        if not greater_better and not (eval_rs_value == 0.0).all():
            gen_eval_rs_value = 1 - gen_eval_rs_value

        v.append(
            [[v[2][i][0], gen_eval_rs_value[i]] for i in range(len(gen_eval_rs_value))]
        )

    for k in removed:
        aggregated_eval_rs_map.pop(k)
    return aggregated_eval_rs_map


def read_eval_rs(settings, target_score, case, model_key, dataset_key):
    target_score_path = os.path.join(
        ".", "evaluation_result", target_score, case, model_key, dataset_key
    )
    rs = {}
    keys = os.listdir(target_score_path)
    for key in keys:
        instance_path = os.path.join(target_score_path, key)
        if os.path.isdir(instance_path):
            setting_key = key
            if target_score == "sa":
                setting = settings.sa_settings[setting_key]
                exp = settings.sa_settings[setting_key].get("exp")
                des = settings.sa_settings[setting_key]["description"]
                name = settings.sa_settings[setting_key]["name"]

            try:
                local_heat_mean = data_utils.read_array(
                    instance_path, "local_heat_mean"
                )
                local_heat_sum = data_utils.read_array(instance_path, "local_heat_sum")
                overall_heat_mean = data_utils.read_array(
                    instance_path, "overall_heat_mean"
                )
                overall_heat_sum = data_utils.read_array(
                    instance_path, "overall_heat_sum"
                )
                original_pred_score = data_utils.read_array(
                    instance_path, "original_pred_score"
                )
                recovered_pred_score = data_utils.read_array(
                    instance_path, "recovered_pred_score"
                )
                original_pred_prob = data_utils.read_array(
                    instance_path, "original_pred_prob"
                )
                recovered_pred_prob = data_utils.read_array(
                    instance_path, "recovered_pred_prob"
                )
                dauc_prob = data_utils.read_array(instance_path, "dauc_prob")
                iauc_prob = data_utils.read_array(instance_path, "iauc_prob")
                mprt = data_utils.read_array(instance_path, "mprt")
                entropy = data_utils.read_array(instance_path, "entropy")

                with open(os.path.join(instance_path, f"stat.json")) as f:
                    stat = json.load(f)

                recovered_pred = (
                    local_heat_mean,
                    local_heat_sum,
                    overall_heat_mean,
                    overall_heat_sum,
                    original_pred_score,
                    recovered_pred_score,
                    original_pred_prob,
                    recovered_pred_prob,
                    None,  # no recovered_img here
                )

                rs[name] = [
                    setting_key,
                    name,
                    des,
                    stat,
                    dict(
                        recovered_pred=recovered_pred,
                        dauc_prob=dauc_prob,
                        iauc_prob=iauc_prob,
                        mprt=mprt,
                    ),
                    entropy,
                    setting,
                    instance_path,
                ]
            except Exception as e:
                traceback.print_exc()

    return rs, settings


def process_eval_rs(
    rs,
    settings,
    k="sa",
    batch_size=256,
    force=False,
    include=None,
    use_include_order=False,
    exclude=[],
    seg=False,
    show_only_exist=False,
    debug=False,
):
    eval_rs_map = {}
    quantus_rs_map = {}
    ei = {"columns": ["Function", "Description"], "data": {}}
    rs = rs.copy()
    for kkk in list(rs.keys()):
        if (include is not None and kkk not in include) or (kkk in exclude):
            del rs[kkk]
    time_map = {}
    with tqdm(total=len(rs.keys()), position=0, leave=True) as opbar:
        for setting_key, sa_rss in rs.items():
            key, name, des, stat, eval_rs, entropy, setting, instance_path = sa_rss

            recovered_pred = eval_rs["recovered_pred"]
            quantus_eval_rs = eval_rs["quantus_eval_rs"]
            dauc_prob = eval_rs["dauc_prob"]
            iauc_prob = eval_rs["iauc_prob"]

            opbar.set_postfix({"name": name})
            ext = ".npy"
            if os.path.exists(os.path.join(instance_path, "maps.npy")):
                ext = ".npy"
            if os.path.exists(os.path.join(instance_path, "maps.npz")):
                ext = ".npz"

            rs_path = os.path.join(instance_path, f"eval_rs{ext}")

            if show_only_exist and not os.path.exists(rs_path):
                continue

            if not os.path.exists(rs_path) or force:
                data_utils.set_startime("map read", time_map)
                with open(os.path.join(instance_path, f"maps{ext}"), "rb") as f:
                    maps = data_utils.read_array(instance_path, "maps")
                data_utils.log_end_time("map read", time_map)

                model_key, dataset_key = setting.get("model_key"), setting.get(
                    "dataset_key"
                )
                model = settings.get_model(model_key)
                val_dataset = settings.get_dataset(dataset_key).val_dataset

                if seg:
                    val_seg_dataset = (
                        settings.get_dataset(dataset_key).val_seg_dataset
                        if hasattr(settings.get_dataset(dataset_key), "val_seg_dataset")
                        else None
                    )
                else:
                    val_seg_dataset = None

                (
                    local_heat_mean,
                    local_heat_sum,
                    overall_heat_mean,
                    overall_heat_sum,
                    original_pred_score,
                    recovered_pred_score,
                    original_pred_prob,
                    recovered_pred_prob,
                    recovered_imgs,
                ) = recovered_pred

                l = maps.shape[0]
                all_idx = [i for i in range(l)]
                s = 0
                c = 0
                c_l = int((l / batch_size) * 0.1)
                c_l = c_l if c_l > 0 else l
                # cuda no better than this
                device = torch.device("cpu")
                all_eval_rs = {}
                with tqdm(total=l, position=1, leave=True, desc="------>") as pbar:
                    while s < l:
                        data_utils.set_startime("eval step", time_map)
                        c += 1
                        e = s + batch_size if s + batch_size < l else l
                        data_utils.set_startime("get seg", time_map)
                        image_segs, targets = data_utils.get_images_targets(
                            val_seg_dataset, device, all_idx[s:e]
                        )
                        data_utils.log_end_time("get seg", time_map)
                        # print(maps[all_idx[s:e]].shape)
                        # print(image_segs.shape)
                        data_utils.set_startime("get sal", time_map)
                        saliency_maps = torch.tensor(maps[all_idx[s:e]], device=device)
                        eval_rs = {}
                        recovered_pred = (
                            original_pred_score[all_idx[s:e]],
                            recovered_pred_score[all_idx[s:e]],
                            original_pred_prob[all_idx[s:e]],
                            recovered_pred_prob[all_idx[s:e]],
                            local_heat_mean[all_idx[s:e]],
                            local_heat_sum[all_idx[s:e]],
                            overall_heat_mean[all_idx[s:e]],
                            overall_heat_sum[all_idx[s:e]],
                            None,  # no recovered_img here
                        )
                        data_utils.log_end_time("get sal", time_map)

                        data_utils.set_startime("loss", time_map)
                        eval_rs.update(get_loss(saliency_maps, image_segs))
                        data_utils.log_end_time("loss", time_map)
                        # data_utils.set_startime('pg', time_map)
                        # eval_rs.update(get_pg_score(saliency_maps, image_segs))
                        # data_utils.log_end_time('pg', time_map)
                        data_utils.set_startime("rcap", time_map)
                        eval_rs.update(get_rcap_score(recovered_pred, debug=False))
                        data_utils.log_end_time("rcap", time_map)

                        data_utils.set_startime("aggregate", time_map)
                        for kk, vv in eval_rs.items():
                            if all_eval_rs.get(kk) is None:
                                all_eval_rs[kk] = []
                            all_eval_rs[kk].extend(vv.tolist())
                        data_utils.log_end_time("aggregate", time_map)

                        pbar.update(e - s)
                        s += batch_size
                        if c % c_l == 0:
                            data_utils.set_startime("gc", time_map)
                            gc.collect()
                            data_utils.log_end_time("gc", time_map)

                        data_utils.log_end_time("eval step", time_map)

                if dauc_prob is not None:
                    auc_score = get_auc_score(dauc_prob, iauc_prob)
                    all_eval_rs["DAUC"] = auc_score["DAUC"]
                    all_eval_rs["IAUC"] = auc_score["IAUC"]
                    all_eval_rs["Overall_AUC"] = auc_score["Overall_AUC"]

                if ext == ".npy":
                    np.save(rs_path, all_eval_rs)
                else:
                    np.savez_compressed(rs_path, all_eval_rs)

            else:
                # with open(rs_path, 'rb') as f:
                #     all_eval_rs = np.load(f, allow_pickle=True).item()
                all_eval_rs = data_utils.read_array(instance_path, "eval_rs").item()

            ei["data"][name] = [key, des]

            if all_eval_rs.get("DAUC") is None:
                all_eval_rs["DAUC"] = 0
                all_eval_rs["IAUC"] = 0
                all_eval_rs["Overall_AUC"] = 0

            if all_eval_rs.get("PG") is None:
                all_eval_rs["PG"] = [0]

            eval_rs_map[name] = all_eval_rs

            opbar.update(1)
    if debug:
        data_utils.print_time_map(time_map)

    aggregated_eval_rs_map = aggregate_different_eval_rs(eval_rs_map, extra_info=ei)
    return (
        aggregated_eval_rs_map,
        {},
        include if use_include_order else None,
    )
