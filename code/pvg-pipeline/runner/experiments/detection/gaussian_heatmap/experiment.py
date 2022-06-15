# Type checks
from typing import Dict, List, Tuple, Optional, Union

# System imports
import sys
import os

projectroot = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if projectroot not in sys.path:
    sys.path.insert(1, projectroot)

# Required project imports
from constants.neurorx_bravo.indexing import SubjectKeys as SJ_KEYS

# required imports
from runner.experiments.detection.template import DatasetPointTemplate, ExperimentTemplateDetection
from runner.experiments.detection import experiment_setup
import socket
hostname = socket.gethostname()

class Hyperparameters:
    workspace: Optional[str] = experiment_setup.workspace  # None links to default comet workspace
    project_name: str = experiment_setup.project_name
    progress_freq: int = 5

    detect_sizes: bool = False
    save_dir: str = None

    batch_size: int = 4
    point_file: str = experiment_setup.point_file
    key_file: str = experiment_setup.key_file
    count_file: str = experiment_setup.count_file
    size_file: str = experiment_setup.size_file

    num_workers: int = 4
    epochs: int = 100
    size = experiment_setup.size

    gaussian_filter: Dict = {'sigma': 1.0, 'truncate': 4.0, 'normalize': False, 'reduce_on': 0, 'reduce_by': 1.0}
    exponential: Dict = {'bias': -4.0}

    detection: Dict = {'threshold': 1e-4, 'max_cost': 6, 'argmax': True,
                       'enable_dilation': True, 'existence_threshold': 0.39,
                       'calibration': False, 'save_dir': None, 'sigma': gaussian_filter['sigma'],
                       'size': 7, 'existence_lb': 0.1, 'per_patient': False, 'per_size': False,
                       'sum_count': False, 'pbcount': True, 'entropy': False, 'ovl': False, 'cc_count': False}

    models: Dict[str, Dict] = {'unet': {'in_channels': 5,
                                        'out_channels': 1,
                                        'filters': 32,
                                        'depth': 5,
                                        'p': 0.1
                                        }
                               }
    optimizers: Dict[str, Dict] = {'unet_adam': {'lr': 1e-4, 'weight_decay': 1e-5, 'eps': 1e-8, 'betas': (0.9, 0.999)}}
    schedulers: Dict[str, Dict] = {}
    criterions: Dict[str, Dict] = {'unet_loss':
                                       {'type': 'MSE'}
                                   }

    metrics: Dict = {'detection': False}

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

    cp_stat: Tuple = ('val_avg_loss', 'min')
    cp_period: int = 1
    cp_patience: int = 1
    cp_save_period: int = 0
    rootdir_args: Dict[str, str] = {'loris_dir': SJ_KEYS.MRI_AND_LABEL}


class DatasetDescription(DatasetPointTemplate):
    def __init__(self):
        super(DatasetDescription, self).__init__(resize=True, point_file=Hyperparameters.point_file,
                                                 key_file=Hyperparameters.key_file, size_file=Hyperparameters.size_file,
                                                 count_file=Hyperparameters.count_file)


class ExperimentDescription(ExperimentTemplateDetection):
    def __init__(self):
        super(ExperimentDescription, self).__init__(Hyperparameters)
