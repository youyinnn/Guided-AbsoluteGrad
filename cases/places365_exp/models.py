import os
import torch
import torchvision.models as models
import pathlib


def get_place365_models(arch='alexnet'):

    current_path = pathlib.Path(__file__).parent.resolve()

    # load the pre-trained weights
    model_name = f'{arch}_places365.pth.tar'
    save_path = os.path.join(current_path, 'pretrained')
    model_file = os.path.join(save_path, model_name)
    if not os.access(model_file, os.W_OK):
        weight_url = 'http://places2.csail.mit.edu/models_places365/' + model_name
        os.system(f'wget {weight_url} -P {save_path}')

    model = models.__dict__[arch](num_classes=365)
    checkpoint = torch.load(
        model_file, map_location=lambda storage, loc: storage)

    state_dict = {str.replace(k, 'module.', ''): v for k,
                  v in checkpoint['state_dict'].items()}
    state_dict = {str.replace(k, 'norm.', 'norm'): v for k,
                  v in state_dict.items()}
    state_dict = {str.replace(k, 'conv.', 'conv'): v for k,
                  v in state_dict.items()}
    state_dict = {str.replace(k, 'normweight', 'norm.weight')
                              : v for k, v in state_dict.items()}
    state_dict = {str.replace(k, 'normrunning', 'norm.running')
                              : v for k, v in state_dict.items()}
    state_dict = {str.replace(k, 'normbias', 'norm.bias')
                              : v for k, v in state_dict.items()}
    state_dict = {str.replace(k, 'convweight', 'conv.weight')
                              : v for k, v in state_dict.items()}

    model.load_state_dict(state_dict)

    return model
