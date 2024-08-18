from cv_exp import gradient, cam, others
from .imagenet_seg import *
from .imagenet_mini import *
import torchvision.models as models

models_map = {
    # "resnet50": models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1),
}


def get_model(key):
    if models_map.get(key) is None:
        if key == "resnet50":
            models_map[key] = models.resnet50(
                weights=models.ResNet50_Weights.IMAGENET1K_V1
            )
    return models_map.get(key)


def get_model_taget_layer(key):
    if key == "resnet50":
        return [get_model(key).layer4]


def dataset_len(k):
    return len(get_dataset(k).val_dataset)


data_set_map = {
    # "imagenet_mini": imagenet_mini.ImageNetMini(),
    # "imagenet_seg": ImageNetSeg(),
    # "imagenet919": get_imagenet_919_for_explanation(split_seed=42),
}


def get_dataset(key):
    if data_set_map.get(key) is None:
        if key == "imagenet_seg":
            data_set_map[key] = ImageNetSeg()

    return data_set_map.get(key)


sa_settings = {
    "1": {
        "name": "Vanilla",
        "description": "pure vanilla",
        "fu": gradient.vanilla_gradient,
        "sa_args": {},
        "batch_size": 100,
    },
    "2": {
        "name": "SmoothGrad",
        "description": "smooth: 015, no abs, no square",
        "fu": gradient.smooth_grad,
        "sa_args": {
            "num_samples": 20,
            "std_spread": 0.15,
            "ifabs": False,
            "ifsquare": False,
        },
        "batch_size": 100,
    },
    "2.narrow": {
        "name": "SmoothGrad narrow",
        "description": "smooth: 015, no abs, no square",
        "fu": gradient.saliency_enhance,
        "sa_args": {
            "fu": gradient.smooth_grad,
            "fu_args": {
                "num_samples": 20,
                "std_spread": 0.15,
                "ifabs": False,
                "ifsquare": False,
            },
            "percentage": 0.5,
            "up_scale": [3, 5],
            "down_scale": [0.2, 0.6],
        },
        "batch_size": 100,
    },
    "2.a": {
        "name": "SmoothGrad-MP",
        "description": "smooth: 015, remain 01, misplaced 015",
        "fu": gradient.misplaced_smooth_grad,
        "sa_args": {
            "num_samples": 20,
            "std_spread": 0.15,
            "remainprecentage": 0.07,
            "mispercentage": 0.1,
        },
        "batch_size": 20,
    },
    "2.b": {
        "name": "Proposed_Guided SmoothGrad",
        "description": "smooth: 015, no abs, x 85",
        "fu": gradient.guided_absolute_grad,
        "sa_args": {
            "num_samples": 20,
            "th": 0.85,
            "ifabs": False,
            "aggregation": "guided",
        },
        "batch_size": 100,
    },
    "2.c": {
        "name": "SmoothGrad-SQ",
        "description": "smooth: 015, no abs, square",
        "fu": gradient.smooth_grad,
        "sa_args": {
            "num_samples": 20,
            "std_spread": 0.15,
            "ifabs": False,
            "ifsquare": True,
        },
        "batch_size": 100,
    },
    "4": {
        "name": "GAG-1",
        "description": "x: abs, mean",
        "fu": gradient.guided_absolute_grad,
        "sa_args": {
            "num_samples": 20,
            "th": None,
            "ifabs": True,
            "aggregation": "mean",
        },
        "batch_size": 100,
    },
    "4.b": {
        "name": "GAG-3",
        "description": "x: abs, 0.85 guided",
        "fu": gradient.guided_absolute_grad,
        "sa_args": {
            "num_samples": 20,
            "th": 0.85,
            "ifabs": True,
            "aggregation": "guided",
        },
        "batch_size": 100,
    },
    "4.c": {
        "name": "GAG-4",
        "description": "x: abs, 0.75 guided",
        "fu": gradient.guided_absolute_grad,
        "sa_args": {
            "num_samples": 20,
            "th": 0.75,
            "ifabs": True,
            "aggregation": "guided",
        },
        "batch_size": 100,
    },
    "4.d": {
        "name": "GAG-5",
        "description": "x: abs, 0.65 guided",
        "fu": gradient.guided_absolute_grad,
        "sa_args": {
            "num_samples": 20,
            "th": 0.65,
            "ifabs": True,
            "aggregation": "guided",
        },
        "batch_size": 100,
    },
    "4.e": {
        "name": "GAG-6",
        "description": "x: abs, 0.55 guided",
        "fu": gradient.guided_absolute_grad,
        "sa_args": {
            "num_samples": 20,
            "th": 0.55,
            "ifabs": True,
            "aggregation": "guided",
        },
        "batch_size": 100,
    },
    "4.f": {
        "name": "GAG-7",
        "description": "x: abs, 0.45 guided",
        "fu": gradient.guided_absolute_grad,
        "sa_args": {
            "num_samples": 20,
            "th": 0.45,
            "ifabs": True,
            "aggregation": "guided",
        },
        "batch_size": 100,
    },
    "4.g": {
        "name": "GAG-8",
        "description": "x: abs, 0.35 guided",
        "fu": gradient.guided_absolute_grad,
        "sa_args": {
            "num_samples": 20,
            "th": 0.35,
            "ifabs": True,
            "aggregation": "guided",
        },
        "batch_size": 100,
    },
    "4.h": {
        "name": "GAG-9",
        "description": "x: abs, 0.25 guided",
        "fu": gradient.guided_absolute_grad,
        "sa_args": {
            "num_samples": 20,
            "th": 0.25,
            "ifabs": True,
            "aggregation": "guided",
        },
        "batch_size": 100,
    },
    "4.i": {
        "name": "GAG-10",
        "description": "x: abs, 0.15 guided",
        "fu": gradient.guided_absolute_grad,
        "sa_args": {
            "num_samples": 20,
            "th": 0.15,
            "ifabs": True,
            "aggregation": "guided",
        },
        "batch_size": 100,
    },
    "4.j": {
        "name": "GAG-11",
        "description": "x: abs, 0.95 guided",
        "fu": gradient.guided_absolute_grad,
        "sa_args": {
            "num_samples": 20,
            "th": 0.95,
            "ifabs": True,
            "aggregation": "guided",
        },
        "batch_size": 100,
    },
    "4.k": {
        "name": "GAG-12",
        "description": "x: abs, 1 guided",
        "fu": gradient.guided_absolute_grad,
        "sa_args": {"num_samples": 20, "th": 1, "ifabs": True, "aggregation": "guided"},
        "batch_size": 100,
    },
    "5": {
        "name": "VarGrad",
        "description": "var: 015, no abs, no squared",
        "fu": gradient.var_grad,
        "sa_args": {
            "num_samples": 20,
            "std_spread": 0.15,
            "ifabs": False,
            "ifsquare": False,
        },
        "batch_size": 100,
    },
    "6": {
        "name": "IG",
        "description": "intgrad: 5 step, 2 trails",
        "fu": gradient.integrated_gradients,
        "sa_args": {"steps": 5, "trials": 2},
        "batch_size": 30,
    },
    "6.a": {
        "name": "Neg IG",
        "description": "intgrad: 5 step, 2 trails",
        "fu": gradient.integrated_gradients,
        "sa_args": {"direction": "negative", "steps": 5, "trials": 2},
        "batch_size": 30,
    },
    "6.b": {
        "name": "Pos IG",
        "description": "intgrad: 5 step, 2 trails",
        "fu": gradient.integrated_gradients,
        "sa_args": {"direction": "positive", "steps": 5, "trials": 2},
        "batch_size": 30,
    },
    "6.c": {
        "name": "Abs IG",
        "description": "intgrad: 5 step, 2 trails",
        "fu": gradient.integrated_gradients,
        "sa_args": {"ifabs": True, "steps": 5, "trials": 2},
        "batch_size": 30,
    },
    "6.d": {
        "name": "Proposed_Guided IG",
        "description": "intgrad: 5 step, 2 trails",
        "fu": gradient.integrated_gradients,
        "sa_args": {"aggregation": "guided", "th": 0.85, "steps": 5, "trials": 2},
        "batch_size": 30,
    },
    "6.e": {
        "name": "Abs Proposed_Guided IG",
        "description": "intgrad: 5 step, 2 trails",
        "fu": gradient.integrated_gradients,
        "sa_args": {
            "aggregation": "guided",
            "th": 0.85,
            "ifabs": True,
        },
        "batch_size": 30,
    },
    "6.m": {
        "name": "IG-MP",
        "description": "intgrad: 5 step, 2 trails",
        "fu": gradient.misplaced_ig,
        "sa_args": {
            "remainprecentage": 0.07,
            "mispercentage": 0.1,
            "steps": 5,
            "trials": 2,
        },
        "batch_size": 30,
    },
    "7": {
        "name": "GB",
        "description": "guidedbp: 1 iteration",
        "fu": gradient.guided_back_propagation,
        "sa_args": {},
        "batch_size": 100,
    },
    "7.a": {
        "name": "Abs GB",
        "description": "guidedbp: 1 iteration",
        "fu": gradient.guided_back_propagation,
        "sa_args": {"ifabs": True},
        "batch_size": 100,
    },
    "7.b": {
        "name": "Neg GB",
        "description": "guidedbp: 1 iteration",
        "fu": gradient.guided_back_propagation,
        "sa_args": {"direction": "negative"},
        "batch_size": 100,
    },
    "7.c": {
        "name": "Both GB",
        "description": "guidedbp: 1 iteration",
        "fu": gradient.guided_back_propagation,
        "sa_args": {"direction": "both"},
        "batch_size": 100,
    },
    "7.m": {
        "name": "GB-MP",
        "description": "guidedbp: 1 iteration",
        "fu": gradient.misplaced_gb,
        "sa_args": {"remainprecentage": 0.03, "mispercentage": 0.1},
        "batch_size": 100,
    },
    "11.a": {
        "name": "BlurIG",
        "description": "blur ig, n 20 r 20",
        "fu": gradient.blur_ig,
        "sa_args": {"num_samples": 20, "radius": 20},
        "batch_size": 10,
    },
    "11.b": {
        "name": "Abs BlurIG",
        "description": "abs blur ig, n 20 r 20",
        "fu": gradient.blur_ig,
        "sa_args": {"num_samples": 20, "radius": 20, "ifabs": True},
        "batch_size": 10,
    },
    "11.c": {
        "name": "Pos BlurIG",
        "description": "+ blur ig, n 20 r 20",
        "fu": gradient.blur_ig,
        "sa_args": {"num_samples": 20, "radius": 20, "direction": "positive"},
        "batch_size": 10,
    },
    "11.d": {
        "name": "Neg BlurIG",
        "description": "- blur ig, n 20 r 20",
        "fu": gradient.blur_ig,
        "sa_args": {"num_samples": 20, "radius": 20, "direction": "negative"},
        "batch_size": 10,
    },
    "11.e": {
        "name": "Proposed_Guided BlurIG",
        "description": "blur ig, n 20 r 20",
        "fu": gradient.blur_ig,
        "sa_args": {
            "num_samples": 20,
            "radius": 20,
            "aggregation": "guided",
            "th": 0.85,
        },
        "batch_size": 10,
    },
    "11.f": {
        "name": "Abs Proposed_Guided BlurIG",
        "description": "blur ig, n 20 r 20",
        "fu": gradient.blur_ig,
        "sa_args": {
            "num_samples": 20,
            "radius": 20,
            "aggregation": "guided",
            "th": 0.85,
            "ifabs": True,
        },
        "batch_size": 10,
    },
    "11.m": {
        "name": "BlurIG-MP",
        "description": "blur ig, n 20 r 20",
        "fu": gradient.misplaced_blurIg,
        "sa_args": {
            "num_samples": 20,
            "radius": 20,
            "remainprecentage": 0.07,
            "mispercentage": 0.1,
        },
        "batch_size": 10,
    },
    "12": {
        "name": "GuidedIG",
        "description": "guided ig, 10 num_samples,",
        "fu": gradient.guided_ig,
        "sa_args": {
            "num_samples": 20,
        },
        "batch_size": 50,
    },
    "12.e": {
        "name": "Proposed_Guided GuidedIG",
        "description": "guided ig, 10 num_samples,",
        "fu": gradient.guided_ig,
        "sa_args": {
            "num_samples": 20,
            "aggregation": "guided",
            "th": 0.85,
        },
        "batch_size": 50,
    },
    "12.f": {
        "name": "Abs Proposed_Guided GuidedIG",
        "description": "guided ig, 10 num_samples,",
        "fu": gradient.guided_ig,
        "sa_args": {
            "num_samples": 20,
            "aggregation": "guided",
            "th": 0.85,
            "ifabs": True,
        },
        "batch_size": 50,
    },
    "12.a": {
        "name": "Abs GuidedIG",
        "description": "guided ig, 10 num_samples,",
        "fu": gradient.guided_ig,
        "sa_args": {"num_samples": 20, "ifabs": True},
        "batch_size": 50,
    },
    "12.b": {
        "name": "Pos GuidedIG",
        "description": "guided ig, 10 num_samples,",
        "fu": gradient.guided_ig,
        "sa_args": {"num_samples": 20, "direction": "positive"},
        "batch_size": 50,
    },
    "12.c": {
        "name": "Neg GuidedIG",
        "description": "guided ig, 10 num_samples,",
        "fu": gradient.guided_ig,
        "sa_args": {"num_samples": 20, "direction": "negative"},
        "batch_size": 50,
    },
    "12.d": {
        "name": "GuidedIG-MP",
        "description": "guided ig, 10 num_samples,",
        "fu": gradient.misplaced_guided_ig,
        "sa_args": {"num_samples": 20, "remainprecentage": 0.07, "mispercentage": 0.1},
        "batch_size": 50,
    },
    "13": {
        "name": "Pos SmoothGrad",
        "description": "pos smooth: 015",
        "fu": gradient.one_direction_smooth_grad,
        "sa_args": {"num_samples": 20, "std_spread": 0.15, "positive_only": True},
        "batch_size": 100,
    },
    "14": {
        "name": "Neg SmoothGrad",
        "description": "neg smooth: 015",
        "fu": gradient.one_direction_smooth_grad,
        "sa_args": {"num_samples": 20, "std_spread": 0.15, "positive_only": False},
        "batch_size": 100,
    },
    "15": {
        "name": "GradCAM",
        "description": "gradcam",
        "fu": cam.grad_cam,
        "sa_args": {},
        "batch_size": 100,
    },
    "16": {
        "name": "Seg",
        "description": "seg",
        "fu": others.segmentation_exp,
        "sa_args": {},
        "batch_size": 100,
    },
}


for k, v in sa_settings.items():
    sa_settings[k]["exp"] = "imagenet"
    sa_settings[k]["model_key"] = sa_settings[k].get("model_key", "resnet50")
    sa_settings[k]["dataset_key"] = sa_settings[k].get("dataset_key", "imagenet_seg")
