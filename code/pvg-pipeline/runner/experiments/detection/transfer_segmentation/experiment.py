 # Type checks
from typing import Dict, Tuple, Optional
import torch

# System imports
import sys
import os

# Add project root to sys.path

projectroot = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if projectroot not in sys.path:
    sys.path.insert(1, projectroot)

# Required project imports
from runner.experiments.detection import experiment_setup
from constants.neurorx_bravo.indexing import SubjectKeys as SJ_KEYS
from constants.pipeline import CheckpointKeys as CP_KEYS
# user imports
from runner.models.nnunet import NNUNet

from runner.experiments.detection.template import DatasetPointSegTemplate, ExperimentSegmentationTemplate
import socket
hostname = socket.gethostname()


class Hyperparameters:
    workspace: Optional[str] = None  # None links to default comet workspace
    project_name: str = 'Point-Detection'  # TODO: change project name appropriately
    progress_freq: int = 5

    detect_sizes: bool = False
    save_dir: str = None

    batch_size: int = 4
    point_file: str = experiment_setup.point_file
    key_file: str = experiment_setup.key_file
    count_file: str = experiment_setup.count_file
    size_file: str = experiment_setup.size_file

    # TODO: fill in pre-trained experiment
    pretrained_unet = 'experiment/checkpoints/experiment_model_best.pth.tar'

    metrics: Dict = {'detection': False}

    size = experiment_setup.size

    num_workers: int = 4
    epochs: int = 60
    resize: bool = False  # resize the input image to (64, 192, 192), necessary for DynUnet
    models: Dict[str, Dict] = {'unet': {'in_channels': 5,
                                        'out_channels': 1,
                                        'filters': 32,
                                        'depth': 5,
                                        'p': 0.1
                                        }
                               }
    optimizers: Dict[str, Dict] = {'unet_adam': {'lr': 1e-4, 'weight_decay': 1e-5, 'eps': 1e-8, 'betas': (0.9, 0.999)}}
    schedulers: Dict[str, Dict] = {}
    criterions: Dict[str, Dict] = {'unet_loss': {'weighted': True, 'decaying': False, 'pos_weight': 3,
                                                 'decay_factor': 0.97}}

    detection: Dict = {'threshold': 0.866,
                       'max_cost': 6, 'argmax': False,
                       'enable_dilation': False, 'existence_threshold': 0.67,
                       'calibration': False, 'save_dir': None, 'per_patient': False,
                       'pbcount': False, 'per_size': True, 'cc_count': False}

    data_augmentations = {'WrappedRandomErasing3D': None,
                          'WrappedRandomCropping3D': None,
                          'WrappedRandomAffine3D': {'degrees': [[-8., 8.], [-8., 8.], [-8., 8.]],
                                                    'translate': [0., 0., 0.], 'scale': [0.90, 1.10],
                                                    'shear': [[-8., 8.], [-8., 8.], [-8, 8.], [-8., 8.], [-8., 8.],
                                                              [-8., 8.]],
                                                    'p': 0.9},
                          'WrappedRandomHorizontalFlip': {'p': 0.5},
                          'WrappedRandomVerticalFlip': {'p': 0.5},
                          'WrappedRandomDepthicalFlip': {'p': 0.5}
                          }

    cp_stat: Tuple = ('val_F1', 'max')
    cp_period: int = 1
    cp_patience: int = 1
    cp_save_period: int = 0
    rootdir_args: Dict[str, str] = {'loris_dir': SJ_KEYS.MRI_AND_LABEL}


class DatasetDescription(DatasetPointSegTemplate):
    def __init__(self):
        super(DatasetDescription, self).__init__(resize=True, point_file=Hyperparameters.point_file,
                                                 key_file=Hyperparameters.key_file, size_file=Hyperparameters.size_file,
                                                 count_file=Hyperparameters.count_file)


class ExperimentDescription(ExperimentSegmentationTemplate):
    def __init__(self):
        super(ExperimentDescription, self).__init__(Hyperparameters)

    def setup_models(self)->None:
        unet = NNUNet(**self.HP.models['unet'])
        model_checkpoint = torch.load(self.HP.pretrained_unet_checkpoint, map_location='cpu')
        unet.load_state_dict(model_checkpoint[CP_KEYS.MODELS]["unet"])
        unet = unet.to(self.devices[0])
        self.models['unet'] = unet

    def setup_optimizers(self)->None:
        params = list(self.models['unet'].decoder.parameters()) + list(self.models['unet'].classifier.parameters())
        self.optimizers['unet_adam'] = torch.optim.Adam(params, **self.HP.optimizers['unet_adam'])
