import pathlib
import os
current_path = pathlib.Path(__file__).parent.resolve()


class IsicAgrs():
    def __init__(self) -> None:
        self.kernel_type = '9c_b4ns_448_ext_15ep-newfold'
        self.out_dim = 9
        self.data_dir = 'data'
        self.data_folder = 512
        self.image_size = 448
        self.use_meta = False
        self.batch_size = 64
        self.num_workers = 0
        self.eval = 'best'
        self.n_meta_dim = '512,128'
        self.model_dir = os.path.join(current_path, 'weights')
        self.enet_type = 'tf_efficientnet_b4_ns'
        self.n_test = 8
