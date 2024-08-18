from .models import *
from .dataset import *
from cv_exp import gradient, cam

models_map = {
    # "effnet": ModelFx(0).model
}


def get_model(key):
    if models_map.get(key) is None:
        if key == 'effnet':
            models_map[key] = ModelFx(0).model
    return models_map.get(key)


data_set_map = {
    # "isic": MelanomaDatasetFx(fold=0),
}


def get_model_taget_layer(key):
    if key == 'effnet':
        return [getattr(get_model(key).enet.blocks, '6')]


def get_dataset(key):
    if data_set_map.get(key) is None:
        if key == 'isic':
            data_set_map[key] = MelanomaDatasetFx(fold=0)

    return data_set_map.get(key)


models_target_layer_map = {}


def dataset_len(k):
    return len(get_dataset(k).val_dataset)


sa_settings = {
    "1_isic": {
        "name": "Vanilla",
        "description": "pure vanilla",
        "fu": gradient.vanilla_gradient,
        "sa_args": {},
        "batch_size": 10,
    },
    "2_isic": {
        "name": "SmoothGrad",
        "description": "smooth: 015, no abs, no square",
                "fu": gradient.smooth_grad,
                "sa_args": {
                    "num_samples": 10, "std_spread": 0.15,
                    "ifabs": False, "ifsquare": False
                },
        "batch_size": 20,
    },
    "2.a_isic": {
        "name": "SmoothGrad-MP",
        "description": "smooth: 015, remain 01, misplaced 015",
                "fu": gradient.misplaced_smooth_grad,
                "sa_args": {
                    "num_samples": 10, "std_spread": 0.15,
                    'remainprecentage': 0.02, 'mispercentage': 0.3
                },
        "batch_size": 20,
    },
    "2.b_isic": {
        "name": "Proposed_Guided SmoothGrad",
        "description": "smooth: 015, no abs, x 85",
                "fu": gradient.guided_absolute_grad,
                "sa_args": {
                    "num_samples": 10, "th": 0.85, "ifabs": False,
                    "aggregation": 'guided'
                },
        "batch_size": 20,
    },
    "2.c_isic": {
        "name": "SmoothGrad-SQ",
        "description": "smooth: 015, no abs, square",
                "fu": gradient.smooth_grad,
                "sa_args": {
                    "num_samples": 10, "std_spread": 0.15,
                    "ifabs": False, "ifsquare": True
                },
        "batch_size": 20,
    },
    "4_isic": {
        "name": "GAG-1",
        "description": "x: abs, mean",
                "fu": gradient.guided_absolute_grad,
                "sa_args": {
                    "num_samples": 10, "th": None, "ifabs": True,
                    "aggregation": 'mean'
                },
        "batch_size": 20,
    },
    "4.b_isic": {
        "name": "GAG-3",
        "description": "x: abs, 0.85 guided",
                "fu": gradient.guided_absolute_grad,
                "sa_args": {
                    "num_samples": 10, "th": 0.85, "ifabs": True,
                    "aggregation": 'guided'
                },
        "batch_size": 20,
    },
    "4.c_isic": {
        "name": "GAG-4",
        "description": "x: abs, 0.75 guided",
                "fu": gradient.guided_absolute_grad,
                "sa_args": {
                    "num_samples": 10, "th": 0.75, "ifabs": True,
                    "aggregation": 'guided'
                },
        "batch_size": 20,
    },
    "4.d_isic": {
        "name": "GAG-5",
        "description": "x: abs, 0.65 guided",
                "fu": gradient.guided_absolute_grad,
                "sa_args": {
                    "num_samples": 10, "th": 0.65, "ifabs": True,
                    "aggregation": 'guided'
                },
        "batch_size": 20,
    },
    "4.e_isic": {
        "name": "GAG-6",
        "description": "x: abs, 0.55 guided",
                "fu": gradient.guided_absolute_grad,
                "sa_args": {
                    "num_samples": 10, "th": 0.55, "ifabs": True,
                    "aggregation": 'guided'
                },
        "batch_size": 20,
    },
    "4.f_isic": {
        "name": "GAG-7",
        "description": "x: abs, 0.45 guided",
                "fu": gradient.guided_absolute_grad,
                "sa_args": {
                    "num_samples": 10, "th": 0.45, "ifabs": True,
                    "aggregation": 'guided'
                },
        "batch_size": 20,
    },
    "4.g_isic": {
        "name": "GAG-8",
        "description": "x: abs, 0.35 guided",
                "fu": gradient.guided_absolute_grad,
                "sa_args": {
                    "num_samples": 10, "th": 0.35, "ifabs": True,
                    "aggregation": 'guided'
                },
        "batch_size": 20,
    },
    "4.h_isic": {
        "name": "GAG-9",
        "description": "x: abs, 0.25 guided",
                "fu": gradient.guided_absolute_grad,
                "sa_args": {
                    "num_samples": 10, "th": 0.25, "ifabs": True,
                    "aggregation": 'guided'
                },
        "batch_size": 20,
    },
    "4.i_isic": {
        "name": "GAG-10",
        "description": "x: abs, 0.15 guided",
                "fu": gradient.guided_absolute_grad,
                "sa_args": {
                    "num_samples": 10, "th": 0.15, "ifabs": True,
                    "aggregation": 'guided'
                },
        "batch_size": 20,
    },
    "5_isic": {
        "name": "VarGrad",
        "description": "var: 015, no th, no abs, no squared",
                "fu": gradient.var_grad,
                "sa_args": {
                    "num_samples": 10, "std_spread": 0.15,
                    "ifabs": False, "ifsquare": False
                },
        "batch_size": 20,
    },
    "6_isic": {
        "name": "IG",
        "description": "intgrad: 5 step, 2 trails",
        "fu": gradient.integrated_gradients,
        "sa_args": {
            'steps': 5,
            'trials': 2
        },
        "batch_size": 20,
    },
    "6_isic2": {
        "name": "IG2",
        "description": "intgrad: 5 step, 2 trails",
        "fu": gradient.integrated_gradients,
        "sa_args": {
            'steps': 10,
            'trials': 3
        },
        "batch_size": 20,
    },
    "6.a_isic": {
        "name": "Neg IG",
        "description": "intgrad: 5 step, 2 trails",
        "fu": gradient.integrated_gradients,
        "sa_args": {
            'direction': 'negative',
            'steps': 5,
            'trials': 2
        },
        "batch_size": 20,
    },
    "6.a_isic2": {
        "name": "Neg IG2",
        "description": "intgrad: 5 step, 2 trails",
        "fu": gradient.integrated_gradients,
        "sa_args": {
            'direction': 'negative',
            'steps': 5,
            'trials': 2
        },
        "batch_size": 20,
    },
    "6.b_isic": {
        "name": "Pos IG",
        "description": "intgrad: 5 step, 2 trails",
        "fu": gradient.integrated_gradients,
        "sa_args": {
            'direction': 'positive',
            'steps': 5,
            'trials': 2
        },
        "batch_size": 20,
    },
    "6.b_isic2": {
        "name": "Pos IG2",
        "description": "intgrad: 5 step, 2 trails",
        "fu": gradient.integrated_gradients,
        "sa_args": {
            'direction': 'positive',
            'steps': 10,
            'trials': 3
        },
        "batch_size": 20,
    },
    "6.c_isic": {
        "name": "Abs IG",
        "description": "intgrad: 5 step, 2 trails",
        "fu": gradient.integrated_gradients,
        "sa_args": {
            'ifabs': True,
            'steps': 10,
            'trials': 3
        },
        "batch_size": 20,
    },
    "6.c_isic2": {
        "name": "Abs IG2",
        "description": "intgrad: 10 step, 3 trails",
        "fu": gradient.integrated_gradients,
        "sa_args": {
            'ifabs': True,
            'steps': 10,
            'trials': 3
        },
        "batch_size": 20,
    },
    "6.d_isic": {
        "name": "Proposed_Guided IG",
        "description": "intgrad: 5 step, 2 trails",
        "fu": gradient.integrated_gradients,
        "sa_args": {
            'aggregation': 'guided',
            'th': 0.85,
            'steps': 5,
            'trials': 2
        },
        "batch_size": 20,
    },
    "6.d_isic2": {
        "name": "Proposed_Guided IG2",
        "description": "intgrad: 5 step, 2 trails",
        "fu": gradient.integrated_gradients,
        "sa_args": {
            'aggregation': 'guided',
            'th': 0.85,
            'steps': 10,
            'trials': 3
        },
        "batch_size": 20,
    },
    "6.e_isic": {
        "name": "Abs Proposed_Guided IG",
        "description": "intgrad: 5 step, 2 trails",
        "fu": gradient.integrated_gradients,
        "sa_args": {
            'aggregation': 'guided',
            'th': 0.85,
            'ifabs': True,
            'steps': 5,
            'trials': 2
        },
        "batch_size": 20,
    },
    "6.e_isic2": {
        "name": "Abs Proposed_Guided IG2",
        "description": "intgrad: 5 step, 2 trails",
        "fu": gradient.integrated_gradients,
        "sa_args": {
            'aggregation': 'guided',
            'th': 0.85,
            'ifabs': True,
            'steps': 10,
            'trials': 3
        },
        "batch_size": 20,
    },
    "6.m_isic": {
        "name": "IG-MP",
        "description": "intgrad: 5 step, 2 trails",
        "fu": gradient.misplaced_ig,
        "sa_args": {
            'remainprecentage': 0.01, 'mispercentage': 0.1,
            'steps': 5,
            'trials': 2
        },
        "batch_size": 20,
    },
    "6.m_isic2": {
        "name": "IG-MP2",
        "description": "intgrad: 5 step, 2 trails",
        "fu": gradient.misplaced_ig,
        "sa_args": {
            'remainprecentage': 0.01, 'mispercentage': 0.1,
            'steps': 10,
            'trials': 3
        },
        "batch_size": 20,
    },
    "7_isic": {
        "name": "GB",
        "description": "guidedbp: 1 iteration",
                "fu": gradient.guided_back_propagation,
                "sa_args": {},
        "batch_size": 10,
    },
    "7.a_isic": {
        "name": "Abs GB",
        "description": "guidedbp: 1 iteration",
                "fu": gradient.guided_back_propagation,
                "sa_args": {
                    'ifabs': True
                },
        "batch_size": 10,
    },
    "7.b_isic": {
        "name": "Neg GB",
        "description": "guidedbp: 1 iteration",
                "fu": gradient.guided_back_propagation,
                "sa_args": {
                    'direction': 'negative'
                },
        "batch_size": 10,
    },
    "7.c_isic": {
        "name": "Both GB",
        "description": "guidedbp: 1 iteration",
                "fu": gradient.guided_back_propagation,
                "sa_args": {
                    'direction': 'both'
                },
        "batch_size": 10,
    },
    # GB = VG, so we use smaller remainprecentage
    "7.m_isic": {
        "name": "GB-MP",
        "description": "guidedbp: 1 iteration",
                "fu": gradient.misplaced_gb,
                "sa_args": {
                    'remainprecentage': 0.005, 'mispercentage': 0.1
                },
        "batch_size": 10,
    },
    "11.a_isic": {
        "name": "BlurIG",
        "description": "blur ig, n 20 r 20",
                "fu": gradient.blur_ig,
                "sa_args": {
                    "num_samples": 20,
                    'radius': 20
                },
        "batch_size": 5,
    },
    "11.b_isic": {
        "name": "Abs BlurIG",
        "description": "abs blur ig, n 20 r 20",
                "fu": gradient.blur_ig,
                "sa_args": {
                    "num_samples": 20,
                    'radius': 20,
                    'ifabs': True
                },
        "batch_size": 5,
    },
    "11.c_isic": {
        "name": "Pos BlurIG",
        "description": "+ blur ig, n 20 r 20",
                "fu": gradient.blur_ig,
                "sa_args": {
                    "num_samples": 20,
                    'radius': 20,
                    'direction': 'positive'
                },
        "batch_size": 5,
    },
    "11.d_isic": {
        "name": "Neg BlurIG",
        "description": "_ blur ig, n 20 r 20",
                "fu": gradient.blur_ig,
                "sa_args": {
                    "num_samples": 20,
                    'radius': 20,
                    'direction': 'negative'
                },
        "batch_size": 5,
    },
    "11.e_isic": {
        "name": "Proposed_Guided BlurIG",
        "description": "blur ig, n 20 r 20",
                "fu": gradient.blur_ig,
                "sa_args": {
                    "num_samples": 20,
                    'radius': 20,
                    'aggregation': 'guided',
                    'th': 0.85
                },
        "batch_size": 5,
    },
    "11.f_isic": {
        "name": "Abs Proposed_Guided BlurIG",
        "description": "blur ig, n 20 r 20",
                "fu": gradient.blur_ig,
                "sa_args": {
                    "num_samples": 20,
                    'radius': 20,
                    'aggregation': 'guided',
                    'th': 0.85,
                    'ifabs': True
                },
        "batch_size": 5,
    },
    "11.m_isic": {
        "name": "BlurIG-MP",
        "description": "blur ig, n 20 r 20",
                "fu": gradient.misplaced_blurIg,
                "sa_args": {
                    "num_samples": 20,
                    'radius': 20,
                    'remainprecentage': 0.02, 'mispercentage': 0.3
                },
        "batch_size": 10,
    },
    "12_isic": {
        "name": "GuidedIG",
        "description": "guided ig, 10 num_samples,",
                "fu": gradient.guided_ig,
                "sa_args": {
                    "num_samples": 10,
                },
        "batch_size": 10,
    },
    "12.a_isic": {
        "name": "Abs GuidedIG",
        "description": "guided ig, 10 num_samples,",
                "fu": gradient.guided_ig,
                "sa_args": {
                    "num_samples": 10,
                    'ifabs': True
                },
        "batch_size": 10,
    },
    "12.b_isic": {
        "name": "Pos GuidedIG",
        "description": "guided ig, 10 num_samples,",
                "fu": gradient.guided_ig,
                "sa_args": {
                    "num_samples": 10,
                    'direction': 'positive'
                },
        "batch_size": 10,
    },
    "12.c_isic": {
        "name": "Neg GuidedIG",
        "description": "guided ig, 10 num_samples,",
                "fu": gradient.guided_ig,
                "sa_args": {
                    "num_samples": 10,
                    'direction': 'negative'
                },
        "batch_size": 10,
    },
    "12.d_isic": {
        "name": "GuidedIG-MP",
        "description": "guided ig, 10 num_samples,",
                "fu": gradient.misplaced_guided_ig,
                "sa_args": {
                    "num_samples": 10,
                    'remainprecentage': 0.02, 'mispercentage': 0.3
                },
        "batch_size": 10,
    },
    "12.e_isic": {
        "name": "Proposed_Guided GuidedIG",
        "description": "guided ig, 10 num_samples,",
                "fu": gradient.guided_ig,
                "sa_args": {
                    "num_samples": 10,
                    'aggregation': 'guided',
                    'th': 0.85,
                },
        "batch_size": 10,
    },
    "12.f_isic": {
        "name": "Abs Proposed_Guided GuidedIG",
        "description": "guided ig, 10 num_samples,",
                "fu": gradient.guided_ig,
                "sa_args": {
                    "num_samples": 10,
                    'aggregation': 'guided',
                    'th': 0.85,
                    'ifabs': True
                },
        "batch_size": 10,
    },
    "13_isic": {
        "name": "Pos SmoothGrad",
        "description": "pos smooth: 015",
                "fu": gradient.one_direction_smooth_grad,
                "sa_args": {
                    "num_samples": 10, "std_spread": 0.15,
                    'positive_only': True
                },
        "batch_size": 10,
    },
    "14_isic": {
        "name": "Neg SmoothGrad",
        "description": "neg smooth: 015",
                "fu": gradient.one_direction_smooth_grad,
                "sa_args": {
                    "num_samples": 10, "std_spread": 0.15,
                    'positive_only': False
                },
        "batch_size": 10,
    },
    "15_isic": {
        "name": "GradCAM",
        "description": "gradcam",
        "fu": cam.grad_cam,
        "sa_args": {
        },
        "batch_size": 100,
    },
}


for k, v in sa_settings.items():
    sa_settings[k]['exp'] = 'isic'
    sa_settings[k]['model_key'] = sa_settings[k].get(
        'model_key', 'effnet')
    sa_settings[k]['dataset_key'] = sa_settings[k].get(
        'dataset_key', 'isic')
