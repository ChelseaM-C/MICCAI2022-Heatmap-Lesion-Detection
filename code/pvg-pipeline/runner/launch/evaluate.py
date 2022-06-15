# System imports
import sys
import os
import argparse

# Add project root to sys.path
projectroot = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
if projectroot not in sys.path:
    sys.path.insert(1, projectroot)

# components
from runner.utilities.io import IO
from runner.workflow.logger import MetricCSVLogger, MetricConsoleLogger
from runner.workflow.logger import EpochProgressLogger, TimeConsoleLogger
from runner.workflow.logger import CometMLLogger
from runner.workflow.engine import EngineUtility
from runner.workflow.checkpointer import Checkpoint

# assembly
import torch
from torch.utils.data import Dataset
from typing import Tuple

# data
from torch.utils.data import DataLoader

def add_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e','--experiment', type=str, required=True, help='Path to directory containing experiment description and split')
    parser.add_argument('--disable-comet', action="store_true", required=False, default=False, help='Disables comet.ml logging')
    parser.add_argument('-g','--gpu', required=False, nargs='+', default=[], help='Specify set of gpus')
    parser.add_argument('-v', '--validate', action="store_true", required=False, default=False, help='Specify whether to use validation set or test set')
    return parser

# core logic
def main():

    # parse and setup args
    parser = add_argparser()
    args, _ = parser.parse_known_args()
    if not os.path.exists(args.experiment):
        raise Exception("Experiment directory does not exist")
    if not os.path.isdir(args.experiment):
        raise Exception("Experiment path should point to a directory")

    # link to experiment folder
    sys.path.insert(0, args.experiment)
    from experiment import Hyperparameters as HP
    from experiment import ExperimentDescription
    from experiment import DatasetDescription

    splits_keys = ['experiment']
    name = splits_keys[0]

    args = parser.parse_args()

    # setup device
    if len(args.gpu) == 0: devices = None
    else:
        devices = [torch.device(f'cuda:{gpu}') for gpu in args.gpu]
        torch.cuda.set_device(devices[0])

    folders = IO.setup_folders(splits_keys, args.experiment)

    # get folders
    metric_folder, checkpoint_folder = folders[name]

    # set up experiment descriptors
    experiment_descriptor = ExperimentDescription()

    # setup experiment objects
    experiment_descriptor.setup_devices(devices)
    experiment_descriptor.setup_objects()
    if args.validate:
        experiment_descriptor.setup_evaluate_handlers()
        experiment_descriptor.setup_evaluate_metrics()
    else:
        experiment_descriptor.setup_test_handlers()
        experiment_descriptor.setup_test_metrics()

    # core components
    models = experiment_descriptor.get_models()

    # process metrics and check safety
    if args.validate:
        packed_test_metrics = experiment_descriptor.get_evaluate_metrics()
    else:
        packed_test_metrics = experiment_descriptor.get_test_metrics()
    if packed_test_metrics is not None:
        test_metrics, test_metric_names = packed_test_metrics
    else: raise ValueError('At least one metric must be defined for testing')

    # process metrics and check safety
    if args.validate:
        packed_test_handlers = experiment_descriptor.get_evaluate_handlers()
    else:
        packed_test_handlers = experiment_descriptor.get_test_handlers()
    if packed_test_handlers is not None:
        assert isinstance(packed_test_handlers, Tuple), "Test Handlers must be type Tuple or None"
        test_handlers, test_handler_states = packed_test_handlers
    else: test_handlers, test_handler_states = [], []

    # TODO: dataset functions (should be done from training)
    dataset_descriptor = DatasetDescription()
    dataset_descriptor.setup_datasets()

    # TODO: Test or Validation Datasets
    if args.validate:
        test_set = dataset_descriptor.get_val_dataset()
        dataset = "Validation"
    else:
        test_set = dataset_descriptor.get_test_dataset()
        dataset = "Test"

    # Dataset safety
    assert isinstance(test_set, Dataset), "Test dataset must be a Dataset object"

    # Create data loaders
    test_loader = DataLoader(test_set, batch_size=HP.batch_size, num_workers=HP.num_workers, pin_memory=False, shuffle=False)

    # create default loggers and timers
    test_progress_logger = EpochProgressLogger(progress_freq=HP.progress_freq, header=f"{name} - {dataset} Progress")
    test_csv_logger = MetricCSVLogger(os.path.join(metric_folder, f'{dataset}_metrics.csv'), HP.cp_period, len(test_loader))
    test_console_logger = MetricConsoleLogger(header=f"{name} - {dataset} Metrics")
    test_time_logger = TimeConsoleLogger(header=f"{name} - {dataset} Times")

    # assemble test engine and bind user metrics and handlers
    if args.validate:
        tester = EngineUtility.assemble(experiment_descriptor.evaluate_step, test_metrics, test_metric_names,
                                        test_handlers, test_handler_states)
    else:
        tester = EngineUtility.assemble(experiment_descriptor.test_step, test_metrics, test_metric_names, test_handlers, test_handler_states)

    # prepare default test loggers
    test_progress_logger.attach(tester)
    test_csv_logger.attach(tester)
    test_console_logger.attach(tester)
    test_time_logger.attach(tester)

    # load model weights with appropriate structure
    model_checkpoint_path = os.path.join(checkpoint_folder, name + '_model_best.pth.tar')
    Checkpoint.load_models(model_checkpoint_path, models)

    # Comet ML Logger
    if not args.disable_comet:
        HP_fields = {k:v for k,v in vars(HP).items() if '__' not in k }
        exp_name = f"{os.path.basename(os.path.abspath(args.experiment))}_{name}"
        cometml_logger = CometMLLogger(project_name=HP.project_name, exp_name=exp_name,
                exp_dir=args.experiment,
                exp_config=HP_fields, checkpoint_dir=checkpoint_folder,
                is_resuming=True, epoch_length=len(test_loader),
                workspace=HP.workspace)

        cometml_logger.attach(tester)

    print("Starting test protocol...")
    torch.backends.cudnn.benchmark = True
    tester.run(test_loader)

if __name__ == '__main__':
    main()