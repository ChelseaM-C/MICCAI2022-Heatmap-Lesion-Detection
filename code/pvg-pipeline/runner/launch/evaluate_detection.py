# System imports
import pickle
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
from runner.metrics.metric import DetectionConfusionMatrix

# assembly
import torch
from torch.utils.data import Dataset
from typing import Tuple
import numpy as np

# data
from torch.utils.data import DataLoader

def add_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e', '--experiment', type=str, required=True,
                        help='Path to directory containing experiment description and split')
    parser.add_argument('--disable-comet', action="store_true", required=False, default=False,
                        help='Disables comet.ml logging')
    parser.add_argument('-g', '--gpu', required=False, nargs='+', default=[], help='Specify set of gpus')
    parser.add_argument('-v', '--validate', action="store_true", required=False, default=False,
                        help='Specify whether to use validation set or test set')
    parser.add_argument('--train', action="store_true", required=False, default=False, help='Specify whether to use train set')

    parser.add_argument('--calibrate', action="store_true", required=False, default=False, help='Run calibration code')
    parser.add_argument('--threshold', type=float, required=False, help='Threshold for F1 - if negative a list is used')
    parser.add_argument('--count', action="store_true", required=False, default=False, help='Run count code')
    parser.add_argument('--size', action="store_true", required=False, default=False, help='Run size code')
    parser.add_argument('--entropy', action="store_true", required=False, default=False, help='Run entropy code')
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
    HP.detection['save_dir'] = args.experiment + 'experiment/metrics/detection'
    if args.train:
        print("Using Train")
        HP.detection['save_dir'] = HP.detection['save_dir'] + "_train"
    elif args.validate:
        print("Using Validation")
        HP.detection['save_dir'] = HP.detection['save_dir'] + "_val"
    else:
        print("Using Test")
        HP.detection['save_dir'] = HP.detection['save_dir'] + "_test"

    if args.calibrate:
        HP.detection['calibration'] = True
    elif args.threshold >= 0.0:
        HP.detection['existence_threshold'] = args.threshold
    else:
        HP.detection['existence_threshold'] = np.arange(0.0, 1.0, 0.01).tolist()

    if args.count:
        HP.detection['sum_count'] = True
        HP.detection['pbcount'] = True

    if args.size:
        HP.detection['per_size'] = True

    if args.entropy:
        HP.detection['entropy'] = True

    if not "entropy" in HP.detection:
        HP.detection['entropy'] = False

    HP.metrics['detection'] = True

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

    if HP.detection['save_dir'] is not None:
        if not args.disable_comet:
            if args.train:
                name = "train"
            elif args.validate:
                name = "val"
            else:
                name = "test"
                stats = pickle.load(open(HP.detection['save_dir']+'_statistics.pk', 'rb'))
                print(stats)
                for thresh in stats:
                    print("f1", 2 * stats[thresh]['ntp'] / (
                                2 * stats[thresh]['ntp'] + stats[thresh]['nfn'] + stats[thresh]['nfp']))
                    print("recall:", stats[thresh]['ntp'] / (stats[thresh]['ntp'] + stats[thresh]['nfn']))
                    print("precision:", stats[thresh]['ntp'] / (stats[thresh]['ntp'] + stats[thresh]['nfp']))
                    print("jaccard:", stats[thresh]['ntp'] / (
                                stats[thresh]['ntp'] + stats[thresh]['nfn'] + stats[thresh]['nfp']))

            if args.size:
                # size
                stats =pickle.load(open(HP.detection['save_dir']+'_statistics_per_size.pk', 'rb'))
                print(stats)

            if HP.detection['cc_count']:
                # count
                stats = pickle.load(open(
                    HP.detection['save_dir']+'_statistics_cc_counts.pk',
                    'rb'))
                acc_sum = 0
                acc_sum_binned = 0
                acc_pb = 0
                total = len(stats)  # / 2
                print(total)
                for i, s in enumerate(stats):
                    gt = s['gt']
                    if True or i % 2 == 0:
                        p = np.round(s['predicted'])
                        if p == gt:
                            acc_sum += 1
                        if gt > 4:
                            gt = 4
                        if p > 4:
                            p = 4
                        if p == gt:
                            acc_sum_binned += 1
                    else:
                        if gt > 4:
                            gt = 4
                        p = np.argmax(s['predicted'][0])
                        if p == gt:
                            acc_pb += 1
                acc_sum /= total
                acc_pb /= total
                acc_sum_binned /= total
                print(acc_sum, acc_pb, acc_sum_binned)

            if args.calibrate:
                DetectionConfusionMatrix.plot_calibraiton_curve(cometml_logger.experiment,
                                                                name, HP.detection['save_dir'])
            if HP.detection['entropy']:
                DetectionConfusionMatrix.plot_entropy_curves(cometml_logger.experiment,
                                                             name, HP.detection['save_dir'])


if __name__ == '__main__':
    main()
