from .models import *
from .dataset import *
from cv_exp import gradient, cam

models_map = {
    # "densenet161": get_place365_models('densenet161')
}


def get_model(key):
    if models_map.get(key) is None:
        if key == 'densenet161':
            models_map[key] = get_place365_models('densenet161')
    return models_map.get(key)


data_set_map = {
    # "places365": Places365()
}


def get_model_taget_layer(key):
    if key == 'densenet161':
        return [get_model(key).features.denseblock4]


def get_dataset(key):
    if data_set_map.get(key) is None:
        if key == 'places365':
            data_set_map[key] = Places365()

    return data_set_map.get(key)


models_target_layer_map = {}


def dataset_len(k):
    return len(get_dataset(k).val_dataset)


sa_settings = {
    "1_places365": {
        "name": "Vanilla",
        "description": "pure vanilla",
                "fu": gradient.vanilla_gradient,
                "sa_args": {},
        "batch_size": 30,
    },
    "2_places365": {
        "name": "SmoothGrad",
        "description": "smooth: 015, no th, no abs, no square",
                "fu": gradient.smooth_grad,
                "sa_args": {
                    "num_samples": 10, "std_spread": 0.15,
                    "ifabs": False, "ifsquare": False
                },
        "batch_size": 20,
    },
    "2.a_places365": {
        "name": "SmoothGrad-MP",
        "description": "smooth: 015, remain 01, misplaced 015",
                "fu": gradient.misplaced_smooth_grad,
                "sa_args": {
                    "num_samples": 10, "std_spread": 0.15,
                    'remainprecentage': 0.07, 'mispercentage': 0.1
                },
        "batch_size": 5,
    },
    "2.b_places365": {
        "name": "Proposed_Guided SmoothGrad",
        "description": "smooth: 015, no abs, x 75",
                "fu": gradient.guided_absolute_grad,
                "sa_args": {
                    "num_samples": 10, "th": 0.75, "ifabs": False,
                    "aggregation": 'guided'
                },
        "batch_size": 20,
    },
    "2.c_places365": {
        "name": "SmoothGrad-SQ",
        "description": "smooth: 015, no abs, square",
                "fu": gradient.smooth_grad,
                "sa_args": {
                    "num_samples": 10, "std_spread": 0.15,
                    "ifabs": False, "ifsquare": True
                },
        "batch_size": 20,
    },
    "4_places365": {
        "name": "GAG-1",
        "description": "x: abs, mean, Uniform",
                "fu": gradient.guided_absolute_grad,
                "sa_args": {
                    "num_samples": 10, "th": None, "ifabs": True,
                    "aggregation": 'mean'
                },
        "batch_size": 20,
    },
    "4.b_places365": {
        "name": "GAG-3",
        "description": "x: abs, 0.85 guided",
                "fu": gradient.guided_absolute_grad,
                "sa_args": {
                    "num_samples": 10, "th": 0.85, "ifabs": True,
                    "aggregation": 'guided'
                },
        "batch_size": 20,
    },
    "4.c_places365": {
        "name": "GAG-4",
        "description": "x: abs, 0.75 guided",
                "fu": gradient.guided_absolute_grad,
                "sa_args": {
                    "num_samples": 10, "th": 0.75, "ifabs": True,
                    "aggregation": 'guided'
                },
        "batch_size": 20,
    },
    "4.d_places365": {
        "name": "GAG-5",
        "description": "x: abs, 0.65 guided",
                "fu": gradient.guided_absolute_grad,
                "sa_args": {
                    "num_samples": 10, "th": 0.65, "ifabs": True,
                    "aggregation": 'guided'
                },
        "batch_size": 20,
    },
    "4.e_places365": {
        "name": "GAG-6",
        "description": "x: abs, 0.55 guided",
                "fu": gradient.guided_absolute_grad,
                "sa_args": {
                    "num_samples": 10, "th": 0.55, "ifabs": True,
                    "aggregation": 'guided'
                },
        "batch_size": 20,
    },
    "4.f_places365": {
        "name": "GAG-7",
        "description": "x: abs, 0.45 guided",
                "fu": gradient.guided_absolute_grad,
                "sa_args": {
                    "num_samples": 10, "th": 0.45, "ifabs": True,
                    "aggregation": 'guided'
                },
        "batch_size": 20,
    },
    "4.g_places365": {
        "name": "GAG-8",
        "description": "x: abs, 0.35 guided",
                "fu": gradient.guided_absolute_grad,
                "sa_args": {
                    "num_samples": 10, "th": 0.35, "ifabs": True,
                    "aggregation": 'guided'
                },
        "batch_size": 20,
    },
    "4.h_places365": {
        "name": "GAG-9",
        "description": "x: abs, 0.25 guided",
                "fu": gradient.guided_absolute_grad,
                "sa_args": {
                    "num_samples": 10, "th": 0.25, "ifabs": True,
                    "aggregation": 'guided'
                },
        "batch_size": 20,
    },
    "4.i_places365": {
        "name": "GAG-10",
        "description": "x: abs, 0.15 guided",
                "fu": gradient.guided_absolute_grad,
                "sa_args": {
                    "num_samples": 10, "th": 0.15, "ifabs": True,
                    "aggregation": 'guided'
                },
        "batch_size": 20,
    },
    "4.j_places365": {
        "name": "GAG-11",
        "description": "x: abs, 0.95 guided",
                "fu": gradient.guided_absolute_grad,
                "sa_args": {
                    "num_samples": 10, "th": 0.95, "ifabs": True,
                    "aggregation": 'guided'
                },
        "batch_size": 20,
    },
    "4.k_places365": {
        "name": "GAG-12",
        "description": "x: abs, 1 guided",
                "fu": gradient.guided_absolute_grad,
                "sa_args": {
                    "num_samples": 10, "th": 1, "ifabs": True,
                    "aggregation": 'guided'
                },
        "batch_size": 20,
    },
    "5_places365": {
        "name": "VarGrad",
        "description": "var: 015, no th, no abs, no squared",
                "fu": gradient.var_grad,
                "sa_args": {
                    "num_samples": 10, "std_spread": 0.15,
                    "ifabs": False, "ifsquare": False
                },
        "batch_size": 20,
    },
    "6_places365": {
        "name": "IG",
        "description": "intgrad: 5 step, 2 trails",
                "fu": gradient.integrated_gradients,
                "sa_args": {
                },
        "batch_size": 10,
    },
    "6.a_places365": {
        "name": "Neg IG",
        "description": "intgrad: 5 step, 2 trails",
                "fu": gradient.integrated_gradients,
                "sa_args": {
                    'direction': 'negative'
                },
        "batch_size": 10,
    },
    "6.b_places365": {
        "name": "Pos IG",
        "description": "intgrad: 5 step, 2 trails",
                "fu": gradient.integrated_gradients,
                "sa_args": {
                    'direction': 'positive'
                },
        "batch_size": 10,
    },
    "6.c_places365": {
        "name": "Abs IG",
        "description": "intgrad: 5 step, 2 trails",
                "fu": gradient.integrated_gradients,
                "sa_args": {
                    'ifabs': True
                },
        "batch_size": 10,
    },
    "6.d_places365": {
        "name": "Proposed_Guided IG",
        "description": "intgrad: 5 step, 2 trails",
                "fu": gradient.integrated_gradients,
                "sa_args": {
                    'aggregation': 'guided',
                    'th': 0.75
                },
        "batch_size": 10,
    },
    "6.e_places365": {
        "name": "Abs Proposed_Guided IG",
        "description": "intgrad: 5 step, 2 trails",
                "fu": gradient.integrated_gradients,
                "sa_args": {
                    'aggregation': 'guided',
                    'th': 0.75,
                    'ifabs': True
                },
        "batch_size": 10,
    },
    "6.m_places365": {
        "name": "IG-MP",
        "description": "intgrad: 5 step, 2 trails",
                "fu": gradient.misplaced_ig,
                "sa_args": {
                    'remainprecentage': 0.07, 'mispercentage': 0.1
                },
        "batch_size": 10,
    },
    "7_places365": {
        "name": "GB",
        "description": "guidedbp: 1 iteration",
                "fu": gradient.guided_back_propagation,
                "sa_args": {},
        "batch_size": 10,
    },
    "7.a_places365": {
        "name": "Abs GB",
        "description": "guidedbp: 1 iteration",
                "fu": gradient.guided_back_propagation,
                "sa_args": {
                    'ifabs': True
                },
        "batch_size": 30,
    },
    "7.b_places365": {
        "name": "Neg GB",
        "description": "guidedbp: 1 iteration",
                "fu": gradient.guided_back_propagation,
                "sa_args": {
                    'direction': 'negative'
                },
        "batch_size": 30,
    },
    "7.c_places365": {
        "name": "Both GB",
        "description": "guidedbp: 1 iteration",
                "fu": gradient.guided_back_propagation,
                "sa_args": {
                    'direction': 'both'
                },
        "batch_size": 30,
    },
    "7.m_places365": {
        "name": "GB-MP",
        "description": "guidedbp: 1 iteration",
                "fu": gradient.misplaced_gb,
                "sa_args": {
                    'remainprecentage': 0.03, 'mispercentage': 0.1
                },
        "batch_size": 30,
    },
    "11.a_places365": {
        "name": "BlurIG",
        "description": "blur ig, n 20 r 20",
                "fu": gradient.blur_ig,
                "sa_args": {
                    "num_samples": 20,
                    'radius': 20
                },
        "batch_size": 10,
    },
    "11.b_places365": {
        "name": "Abs BlurIG",
        "description": "abs blur ig, n 20 r 20",
                "fu": gradient.blur_ig,
                "sa_args": {
                    "num_samples": 20,
                    'radius': 20,
                    'ifabs': True
                },
        "batch_size": 10,
    },
    "11.c_places365": {
        "name": "Pos BlurIG",
        "description": "+blur ig, n 20 r 20",
                "fu": gradient.blur_ig,
                "sa_args": {
                    "num_samples": 20,
                    'radius': 20,
                    'direction': 'positive'
                },
        "batch_size": 10,
    },
    "11.d_places365": {
        "name": "Neg BlurIG",
        "description": "- blur ig, n 20 r 20",
                "fu": gradient.blur_ig,
                "sa_args": {
                    "num_samples": 20,
                    'radius': 20,
                    'direction': 'negative'
                },
        "batch_size": 10,
    },
    "11.e_places365": {
        "name": "Proposed_Guided BlurIG",
        "description": "blur ig, n 20 r 20",
                "fu": gradient.blur_ig,
                "sa_args": {
                    "num_samples": 20,
                    'radius': 20,
                    'aggregation': 'guided',
                    'th': 0.75
                },
        "batch_size": 10,
    },
    "11.f_places365": {
        "name": "Abs Proposed_Guided BlurIG",
        "description": "blur ig, n 20 r 20",
                "fu": gradient.blur_ig,
                "sa_args": {
                    "num_samples": 20,
                    'radius': 20,
                    'aggregation': 'guided',
                    'th': 0.75,
                    'ifabs': True
                },
        "batch_size": 10,
    },
    "11.m_places365": {
        "name": "BlurIG-MP",
        "description": "blur ig, n 20 r 20",
                "fu": gradient.misplaced_blurIg,
                "sa_args": {
                    "num_samples": 20,
                    'radius': 20,
                    'remainprecentage': 0.07, 'mispercentage': 0.1
                },
        "batch_size": 10,
    },
    "12_places365": {
        "name": "GuidedIG",
        "description": "guided ig, 10 num_samples,",
                "fu": gradient.guided_ig,
                "sa_args": {
                    "num_samples": 10,
                },
        "batch_size": 10,
    },
    "12.a_places365": {
        "name": "Abs GuidedIG",
        "description": "guided ig, 10 num_samples,",
                "fu": gradient.guided_ig,
                "sa_args": {
                    "num_samples": 10,
                    'ifabs': True
                },
        "batch_size": 10,
    },
    "12.b_places365": {
        "name": "Pos GuidedIG",
        "description": "guided ig, 10 num_samples,",
                "fu": gradient.guided_ig,
                "sa_args": {
                    "num_samples": 10,
                    'direction': 'positive'
                },
        "batch_size": 10,
    },
    "12.c_places365": {
        "name": "Neg GuidedIG",
        "description": "guided ig, 10 num_samples,",
                "fu": gradient.guided_ig,
                "sa_args": {
                    "num_samples": 5,
                    'direction': 'negative'
                },
        "batch_size": 10,
    },
    "12.d_places365": {
        "name": "GuidedIG-MP",
        "description": "guided ig, 10 num_samples,",
                "fu": gradient.misplaced_guided_ig,
                "sa_args": {
                    "num_samples": 10,
                    'remainprecentage': 0.07, 'mispercentage': 0.1
                },
        "batch_size": 10,
    },
    "12.e_places365": {
        "name": "Proposed_Guided GuidedIG",
        "description": "guided ig, 10 num_samples,",
                "fu": gradient.guided_ig,
                "sa_args": {
                    "num_samples": 10,
                    'aggregation': 'guided',
                    'th': 0.75,
                },
        "batch_size": 10,
    },
    "12.f_places365": {
        "name": "Abs Proposed_Guided GuidedIG",
        "description": "guided ig, 10 num_samples,",
                "fu": gradient.guided_ig,
                "sa_args": {
                    "num_samples": 10,
                    'aggregation': 'guided',
                    'th': 0.75,
                    'ifabs': True
                },
        "batch_size": 10,
    },
    "13_places365": {
        "name": "Pos SmoothGrad",
        "description": "pos smooth: 015",
                "fu": gradient.one_direction_smooth_grad,
                "sa_args": {
                    "num_samples": 10, "std_spread": 0.15,
                    'positive_only': True
                },
        "batch_size": 30,
    },
    "14_places365": {
        "name": "Neg SmoothGrad",
        "description": "neg smooth: 015",
                "fu": gradient.one_direction_smooth_grad,
                "sa_args": {
                    "num_samples": 10, "std_spread": 0.15,
                    'positive_only': False
                },
        "batch_size": 30,
    },
    "15_places365": {
        "name": "GradCAM",
        "description": "gradcam",
        "fu": cam.grad_cam,
        "sa_args": {
        },
        "batch_size": 30,
    },
}


for k, v in sa_settings.items():
    sa_settings[k]['exp'] = 'places365'
    sa_settings[k]['model_key'] = sa_settings[k].get(
        'model_key', 'densenet161')
    sa_settings[k]['dataset_key'] = sa_settings[k].get(
        'dataset_key', 'places365')
