import argparse
import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
import traceback

sys.path.append(os.getcwd())

parser = argparse.ArgumentParser()
parser.add_argument("setting_keys", type=str, help="experiment setting keys", nargs="*")
parser.add_argument(
    "-t",
    "--target",
    type=str,
    help="target experiment of the settings",
    choices=["sa"],
    required=True,
)

parser.add_argument("-b", "--batch_size", type=int, help="batch size", default=None)

parser.add_argument("-stop", "--stop", type=float, help="stop", default=None)

parser.add_argument("-steps", "--steps", type=float, help="steps", default=None)

parser.add_argument("-mk", "--model_key", type=str, help="model key", required=True)

parser.add_argument("-dk", "--dataset_key", type=str, help="dataset key", required=True)

parser.add_argument(
    "-c",
    "--case",
    type=str,
    help="case of the experiment",
    choices=[
        "imagenet",
        "isic",
        "places365",
    ],
    required=True,
)

mode_map = {
    "xai": "xai",
    "x": "xai",
    "eval": "eval",
    "e": "eval",
    "sequence": "sequence",
    "s": "sequence",
    "auc": "auc",
    "a": "auc",
    "sn": "sanity",
    "sanity": "sanity",
}
parser.add_argument(
    "-m",
    "--mode",
    type=str,
    help="xai execution of the given settings",
    choices=mode_map.keys(),
)

args = parser.parse_args()

if __name__ == "__main__":
    mode = mode_map.get(args.mode)
    target_exp = args.target
    model_key = args.model_key
    dataset_key = args.dataset_key
    setting_keys = args.setting_keys
    batch_size = args.batch_size
    stop = args.stop
    steps = args.steps
    case = args.case

    if case in ["imagenet", "imagenet919"]:
        from cases.imagenet_exp import settings
    if case == "isic":
        from cases.isic_exp import settings
    if case == "places365":
        from cases.places365_exp import settings

    from cv_exp.pipe import Pipe
    from cv_exp.eval import batch_rcap, batch_auc, batch_sanity
    from cv_exp.utils import fix_seed, read_array

    fix_seed(42)

    pipe = Pipe()
    device = pipe.device

    if len(setting_keys) == 0:
        setting_keys = settings.sa_settings.keys()

    with tqdm(total=len(setting_keys), position=0, leave=True) as opbar:
        for setting_key in setting_keys:
            opbar.set_postfix({"key": setting_key})
            print(f"Setting Key: {setting_key}")

            # TODO: debug mode
            debug = False

            assert settings.sa_settings.get(setting_key) is not None
            setting = settings.sa_settings[setting_key]

            exts = [".npz", ".npy"]
            # exp = setting['exp']
            current_setting_key = setting_key
            setting_save = os.path.join(
                ".",
                "evaluation_result",
                target_exp,
                args.case,
                model_key,
                dataset_key,
                current_setting_key,
            )

            print(setting_save)

            name = setting.get("name")
            description = setting.get("description")
            func = setting.get("fu")
            sa_args = setting.get("sa_args")
            use_predicted_target = setting.get("use_predicted_target", False)
            start = setting.get("start", 0)
            end = setting.get("end", settings.dataset_len(dataset_key))
            batch_size = (
                batch_size if batch_size is not None else setting.get("batch_size", 100)
            )
            stop = stop if stop is not None else setting.get("stop", 0.5)
            steps = steps if steps is not None else setting.get("steps", 0.1)

            print(f"Setting:\n", setting, batch_size, start, end, stop, steps)

            if mode in ["xai", "sequence"]:
                print("\r\nXAI:")
                # des_file = os.path.join(setting_save, 'des.txt')
                # with open(des_file, 'w') as f:
                #     f.write(description)

                # tracker = EmissionsTracker(
                #     project_name=f'{target_exp}_{current_setting_key}', tracking_mode='machine',
                #     output_dir=setting_save, log_level='critical')
                try:
                    # tracker.start()
                    maps, tt0 = pipe.get_saliency_map(
                        settings,
                        model_key,
                        dataset_key,
                        start,
                        end,
                        batch_size,
                        target_exp,
                        func,
                        sa_args,
                        use_predicted_target,
                        debug=debug,
                    )
                except Exception as e:
                    print(traceback.format_exc())
                    # or
                    print(sys.exc_info()[2])
                    raise e
                finally:
                    # tracker.stop()
                    pass

                if not debug:
                    task_save = os.path.join(setting_save)
                    if not os.path.exists(task_save):
                        os.makedirs(task_save, exist_ok=True)

                    if maps.shape[0] > 0:
                        np.save(os.path.join(task_save, f"maps"), maps)

                    stat = {
                        "n": maps.shape[0],
                        "sa_time": tt0,
                        "name": name,
                        "description": description,
                        "model_key": model_key,
                        "dataset_key": dataset_key,
                        "start": start,
                        "end": end,
                        "batch_size": batch_size,
                        "func": str(func),
                        "sa_args": sa_args,
                    }

                    with open(os.path.join(task_save, "stat.json"), "w") as f:
                        f.write(json.dumps(stat, indent=4, default=lambda o: str(o)))
                else:
                    print(maps.shape)
                    plt.imshow(maps[0])
                    plt.show()

            ext = ".npz"
            for eee in exts:
                if os.path.exists(os.path.join(setting_save, f"maps{eee}")):
                    ext = eee
                    break

            if not os.path.exists(os.path.join(setting_save, f"maps{ext}")):
                print(f"File not exist: {os.path.join(setting_save, f'maps{ext}')}")
                continue

            try:
                os.remove(os.path.join(setting_save, f"eval_rs{ext}"))
            except OSError:
                pass

            if mode in ["eval", "sequence"]:
                print("\r\nRcap Eval:")
                maps = read_array(setting_save, "maps")

                rs = batch_rcap(
                    settings,
                    model_key,
                    dataset_key,
                    start,
                    end,
                    batch_size,
                    maps,
                    device,
                    stop,
                    steps,
                )

                for k, v in rs.items():
                    # print(k, v)
                    np.save(os.path.join(setting_save, f"{k}"), v)

            if mode in ["auc"]:
                print("\r\nD/I AUC Eval:")
                maps = read_array(setting_save, "maps")

                rs = batch_auc(
                    settings,
                    model_key,
                    dataset_key,
                    start,
                    end,
                    batch_size,
                    maps,
                    device,
                )

                for k, v in rs.items():
                    np.save(os.path.join(setting_save, f"{k}"), v)

            if mode == "sanity":
                print("\r\nSanity Check Eval:")
                maps = read_array(setting_save, "maps")

                rs = batch_sanity(
                    settings,
                    model_key,
                    dataset_key,
                    start,
                    end,
                    batch_size,
                    maps,
                    device,
                    func,
                    sa_args,
                )

                for k, v in rs.items():
                    np.save(os.path.join(setting_save, f"{k}"), v)

            opbar.update(1)
            #     del unarys, mean, score_gd_mean, score_gd_max, all_mop
            #     del score_gd_var, score_ld_mean, score_ld_max, score_ld_var
            # del maps

            # torch.cuda.empty_cache()
            # gc.collect()
