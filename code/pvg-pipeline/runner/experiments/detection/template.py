# Type checks
from typing import Callable, Dict, List, Tuple, Union, Optional
from ignite.metrics import Metric
from ignite.engine import Engine, Events
import torch
from torch import device as Device
from torch import Tensor
from torch.optim.lr_scheduler import ExponentialLR
import numpy as np

# System imports
import sys
import os

# Add project root to sys.path
projectroot = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..'))
if projectroot not in sys.path:
    sys.path.insert(1, projectroot)

# Required project imports
from runner.builder.assembly import MetricAssembler, HandlerAssembler

# required imports
import torch.nn as nn

# user imports
from runner.models.nnunet import NNUNet
from torch.optim import Adam
from ignite.metrics import Average
from runner.metrics.metric import BinaryThresholdConfusionMatrix, ApproximateMetrics, btcmF1, DetectionConfusionMatrix, dcmF1
from runner.transforms.augmentation import CustomTransforms
from scipy import ndimage


""" FUNCTIONAL """


def init_weights(m):
    if (type(m) == nn.Conv3d or
        type(m) == nn.ConvTranspose3d):
        nn.init.kaiming_normal_(m.weight)
        if m.bias is not None:
            nn.init.zeros_(m.bias)

""" DATASET """


class DatasetTemplate:
    def __init__(self, resize=False):
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.resize = resize

    def setup_datasets(self):
        pass
        ################## MODIFY HERE #######################
        # TODO: Setup custom dataloaders for train, val, test
        # self.train_dataset =
        # self.val_dataset =
        # self.test_dataset =
        ######################################################

    def get_train_dataset(self):
        return self.train_dataset

    def get_val_dataset(self):
        return self.val_dataset

    def get_test_dataset(self):
        return self.test_dataset

    def set_resize(self, resize):
        self.resize = resize


class DatasetPointTemplate(DatasetTemplate):
    def __init__(self, point_file, key_file, size_file, count_file=None, resize=False):
        super(DatasetPointTemplate, self).__init__(resize)
        self.point_file = point_file
        self.size_file = size_file
        self.key_file = key_file
        self.count_file = count_file

    def setup_datasets(self):
        pass
        ################## MODIFY HERE #######################
        # TODO: Setup custom dataloaders for train, val, test
        # self.train_dataset =
        # self.val_dataset =
        # self.test_dataset =


class DatasetPointSegTemplate(DatasetTemplate):
    def __init__(self, point_file, key_file, size_file, count_file=None, resize=False):
        super(DatasetPointSegTemplate, self).__init__(resize)
        self.point_file = point_file
        self.key_file = key_file
        self.size_file = size_file
        self.count_file = count_file

    def setup_datasets(self):
        pass
        ################## MODIFY HERE #######################
        # TODO: Setup custom dataloaders for train, val, test
        # self.train_dataset =
        # self.val_dataset =
        # self.test_dataset =



""" MSC """


class Exponential(nn.Module):
    def __init__(self, bias):
        super(Exponential, self).__init__()
        self.bias = torch.tensor(bias)

    def forward(self, x):
        return torch.exp(x + self.bias)


""" EXPERIMENT """


class ExperimentTemplate:
    def __init__(self, Hyperparameters):
        self.models = {}
        self.optimizers = {}
        self.schedulers = {}
        self.criterions = {}
        self.HP = Hyperparameters
        self.train_metrics: Optional[Tuple[List[Metric], List[str]]] = None
        self.evaluate_metrics: Optional[Tuple[List[Metric], List[str]]] = None
        self.test_metrics: Optional[Tuple[List[Metric], List[str]]] = None
        self.train_handlers: Optional[Tuple[List[Callable], List[Events]]] = None
        self.evaluate_handlers: Optional[Tuple[List[Callable], List[Events]]] = None
        self.test_handlers: Optional[Tuple[List[Callable], List[Events]]] = None
        self.devices: Optional[List[Device]] = None
        self.augmentations = []

    ############### IMPLEMENT FUNCTIONS BELOW #########################
    def setup_devices(self, devices:Optional[List[Device]]=None)->None:
        if devices is None:
            self.devices = ['cpu']
            # raise Exception("No device received.")
            # Alt option: define CPU device?
        else:
            self.devices = devices

    def setup_models(self)->None:
        unet = NNUNet(**self.HP.models['unet'])
        unet = unet.to(self.devices[0])
        self.models['unet'] = unet

    def setup_optimizers(self)->None:
        self.optimizers['unet_adam'] = Adam(self.models['unet'].parameters(), **self.HP.optimizers['unet_adam'])

    def setup_criterions(self)->None:
        self.criterions['unet_loss'] = self.setup_bce(net='unet_loss')

    def setup_schedulers(self)->None:
        if len(self.HP.schedulers) > 0:
            self.schedulers['unet_scheduler'] = ExponentialLR(self.optimizers['unet_adam'],
                                                              **self.HP.schedulers['unet_scheduler'])

    def setup_objects(self)->None:
        # Models
        self.setup_models()

        # Optimizers
        self.setup_optimizers()

        # Schedulers
        self.setup_schedulers()

        # Criterions
        self.setup_criterions()

        try:
            # Data augmentation
            input_keys = ['MRI', 'SEG_LABEL', 'MASK',
                          # 'POINT_LABEL',
                          'LABEL']
            if self.weighted_loss:
                input_keys.append('LESION_MASK')
            if self.HP.data_augmentations['WrappedRandomErasing3D'] is not None:
                self.augmentations.append(CustomTransforms.WrappedRandomErasing3D(input_mapping=input_keys[:-1],
                                                                                  **self.HP.data_augmentations[
                                                                                      'WrappedRandomErasing3D'],
                                                                                  device=self.devices[0]))
            if self.HP.data_augmentations['WrappedRandomCropping3D'] is not None:
                self.augmentations.append(CustomTransforms.WrappedRandomCropping3D(input_mapping=input_keys,
                                                                                   **self.HP.data_augmentations[
                                                                                       'WrappedRandomCropping3D'],
                                                                                   device=self.devices[0]))
            if self.HP.data_augmentations['WrappedRandomAffine3D'] is not None:
                input_mapping = {'MRI': [1, False], 'LABEL': [0, False],
                                 'MASK': [0, False],
                                  # 'POINT_LABEL': [0, False]
                                 }
                self.augmentations.append(CustomTransforms.WrappedRandomAffine3D(input_mapping=input_mapping,
                                                                                 **self.HP.data_augmentations[
                                                                                     'WrappedRandomAffine3D'],
                                                                                 device=self.devices[0]))
            if self.HP.data_augmentations['WrappedRandomHorizontalFlip'] is not None:
                self.augmentations.append(CustomTransforms.WrappedRandomHorizontalFlip(input_mapping=input_keys,
                                                                                       **self.HP.data_augmentations[
                                                                                           'WrappedRandomHorizontalFlip'],
                                                                                       device=self.devices[0]))
            if self.HP.data_augmentations['WrappedRandomVerticalFlip'] is not None:
                self.augmentations.append(CustomTransforms.WrappedRandomVerticalFlip(input_mapping=input_keys,
                                                                                     **self.HP.data_augmentations[
                                                                                         'WrappedRandomVerticalFlip'],
                                                                                     device=self.devices[0]))
            if 'WrappedRandomDepthicalFlip' in self.HP.data_augmentations:
                if self.HP.data_augmentations['WrappedRandomDepthicalFlip'] is not None:
                    self.augmentations.append(CustomTransforms.WrappedRandomDepthicalFlip(input_mapping=input_keys,
                                                                                           **self.HP.data_augmentations[
                                                                                               'WrappedRandomDepthicalFlip'],
                                                                                           device=self.devices[0]))
        except KeyError or AttributeError:
            print('No data augmentations selected')
            return

    def setup_bce(self, reduction='none'):
        return nn.BCEWithLogitsLoss(reduction=reduction)

    def setup_train_metrics(self)->None:
        assembler = MetricAssembler()

        # Average loss
        avg_loss_metric = Average(output_transform=lambda output: output['loss'], device=self.devices[0])
        assembler.add(avg_loss_metric, 'train_avg_loss')

        # btcm PR AUC, ROC AUC, F1
        btcm = BinaryThresholdConfusionMatrix(thresholds=torch.tensor(self.HP.metric_thresholds), pred_key='preds', label_key='label', device=self.devices[0] )
        PR_AUC = ApproximateMetrics.ApproxPR_AUC(btcm)
        ROC_AUC = ApproximateMetrics.ApproxROC_AUC(btcm)
        F1 = btcmF1(btcm)
        assembler.add(PR_AUC, 'train_PR_AUC')
        assembler.add(ROC_AUC, 'train_ROC_AUC')
        assembler.add(F1, 'train_F1')

        self.train_metrics = assembler.build()

    def setup_evaluate_metrics(self)->None:
        assembler = MetricAssembler()

        # Average loss
        avg_loss_metric = Average(output_transform=lambda output: output['loss'], device=self.devices[0])
        assembler.add(avg_loss_metric, 'val_avg_loss')

        # btcm PR AUC, ROC AUC, F1
        btcm = BinaryThresholdConfusionMatrix(thresholds=torch.tensor(self.HP.metric_thresholds), pred_key='preds', label_key='label', device=self.devices[0] )
        PR_AUC = ApproximateMetrics.ApproxPR_AUC(btcm)
        ROC_AUC = ApproximateMetrics.ApproxROC_AUC(btcm)
        F1 = btcmF1(btcm)
        assembler.add(PR_AUC, 'val_PR_AUC')
        assembler.add(ROC_AUC, 'val_ROC_AUC')
        assembler.add(F1, 'val_F1')

        self.evaluate_metrics = assembler.build()

    def setup_test_metrics(self)->None:
        assembler = MetricAssembler()

        # Average loss
        avg_loss_metric = Average(output_transform=lambda output: output['loss'], device=self.devices[0])
        assembler.add(avg_loss_metric, 'test_avg_loss')

        # btcm PR AUC, ROC AUC, F1
        btcm = BinaryThresholdConfusionMatrix(thresholds=torch.tensor(self.HP.metric_thresholds), pred_key='preds', label_key='label', device=self.devices[0] )
        PR_AUC = ApproximateMetrics.ApproxPR_AUC(btcm)
        ROC_AUC = ApproximateMetrics.ApproxROC_AUC(btcm)
        F1 = btcmF1(btcm)
        assembler.add(PR_AUC, 'test_PR_AUC')
        assembler.add(ROC_AUC, 'test_ROC_AUC')
        assembler.add(F1, 'test_F1')

        self.evaluate_metrics = assembler.build()

    def setup_train_handlers(self) -> None:
        if not len(self.HP.schedulers) > 0:
            return

        # create generic step method
        def scheduler_step(engine):
            self.schedulers['unet_scheduler'].step()

        assembler = HandlerAssembler()
        assembler.add(scheduler_step, Events.EPOCH_COMPLETED)
        self.train_handlers = assembler.build()

    def setup_evaluate_handlers(self)->None:
        pass

    def setup_test_handlers(self)->None:
        pass

    def train_batching(self, batch: Union[Dict, Tensor])->Union[Dict, List, Tensor, Tuple]:
        # Cast labels to float
        batch['LABEL'] = batch['LABEL'].float()

        # Move to GPU
        for key, value in batch.items():
            batch[key] = value.to(self.devices[0], non_blocking=True)

        # Apply transforms
        batch['MRI'] = CustomTransforms.denoise(batch['MRI'], batch['MASK'])
        batch['MRI'] = CustomTransforms.standardize(batch['MRI'], batch['MASK'])

        # Pad the input to be an even size
        if self.HP.resize:
            batch = CustomTransforms.resize(batch, size=(64, 192, 192), imaging_keys=list(batch.keys()))

        # Augment
        for augmentation in self.augmentations:
            batch = augmentation(batch)

        return batch

    def evaluate_batching(self, batch: Union[Dict, Tensor])->Union[Dict, List, Tensor, Tuple]:
        # Cast labels to float
        batch['LABEL'] = batch['LABEL'].float()

        # Move to GPU
        for key, value in batch.items():
            batch[key] = value.to(self.devices[0], non_blocking=True)

        # Apply transforms
        batch['MRI'] = CustomTransforms.denoise(batch['MRI'], batch['MASK'])
        batch['MRI'] = CustomTransforms.standardize(batch['MRI'], batch['MASK'])

        if self.HP.resize:
            batch = CustomTransforms.resize(batch, size=(64, 192, 192), imaging_keys=list(batch.keys()))

        return batch

    def forward(self, batch, mode='train'):
        model = self.models['unet']
        if mode == 'train':
            model.train()
        elif mode == "mc":
            model.eval()
            model.encoder[1].down[1].train()
            model.encoder[2].down[1].train()
            model.encoder[3].down[1].train()
            model.encoder[4].down[1].train()
            model.decoder[0].drop.train()
            model.decoder[1].drop.train()
            model.decoder[2].drop.train()
            model.decoder[3].drop.train()
        else:
            model.eval()
        preds = model(batch['MRI'])
        return preds

    def calculate_losses(self, preds, batch):
        loss = self.criterions['unet_loss'](preds, batch['LABEL'])
        loss = loss[batch['MASK'].bool()].mean()
        return loss

    def backward(self, loss):
        optimizer = self.optimizers['unet_adam']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def prepare_output(self, preds, loss, batch):
        return {'preds': preds[batch['MASK'].bool()].detach(),
                'label': batch['LABEL'][batch['MASK'].bool()].detach(),
                'loss': loss.item()
                }

    def train_step(self, engine: Engine, batch: Union[Dict, Tensor])->Union[Dict, Tensor, None]:
        batch = self.train_batching(batch)

        preds = self.forward(batch, mode='train')

        loss = self.calculate_losses(preds, batch, epoch=engine.state.epoch)

        self.backward(loss)

        return self.prepare_output(preds, loss, batch, epoch=engine.state.epoch)

    @torch.no_grad()
    def evaluate_step(self, engine: Engine, batch: Union[Dict, Tensor])->Union[Dict, Tensor, None]:
        batch = self.evaluate_batching(batch)

        preds = self.forward(batch, mode='eval')

        loss = self.calculate_losses(preds, batch, epoch=engine.state.epoch)

        return self.prepare_output(preds, loss, batch, epoch=engine.state.epoch)

    @torch.no_grad()
    def test_step(self, engine: Engine, batch: Union[Dict, Tensor])->Union[Dict, Tensor, None]:
        batch = self.evaluate_batching(batch)

        preds = self.forward(batch, mode='eval')

        loss = self.calculate_losses(preds, batch, epoch=engine.state.epoch)

        return self.prepare_output(preds, loss, batch, epoch=engine.state.epoch)

    ##################################################################

    def get_models(self):
        return self.models

    def get_optimizers(self):
        return self.optimizers

    def get_schedulers(self):
        return self.schedulers

    def get_criterions(self):
        return self.criterions

    def get_train_metrics(self):
        return self.train_metrics

    def get_evaluate_metrics(self):
        return self.evaluate_metrics

    def get_test_metrics(self):
        return self.test_metrics

    def get_train_handlers(self):
        return self.train_handlers

    def get_evaluate_handlers(self):
        return self.evaluate_handlers

    def get_test_handlers(self):
        return self.test_handlers


class ExperimentTemplateDetection(ExperimentTemplate):
    def __init__(self, Hyperparameters):
        super(ExperimentTemplateDetection, self).__init__(Hyperparameters)
        self.output_function = None

    def setup_objects(self) -> None:
        super().setup_objects()
        self.output_function = Exponential(self.HP.exponential['bias'])

    def setup_criterions(self) -> None:
        if self.HP.criterions['unet_loss']['type'] == 'MSE':
            self.criterions['unet_loss'] = nn.MSELoss(reduction='none')

    def forward(self, batch, mode='train', return_raw=False, epoch=None):
        model = self.models['unet']
        if mode == 'train':
            model.train()
        else:
            model.eval()
        raw_preds = model(batch['MRI'])

        # apply output function
        preds = self.output_function(raw_preds)

        if return_raw:
            return raw_preds, preds
        return None, preds

    def calculate_losses(self, preds_tuple, batch, epoch=None):
        raw_preds, preds = preds_tuple

        loss = self.criterions['unet_loss'](preds, batch['LABEL'])

        loss = loss[batch['MASK'].bool()].sum() / len(batch['LABEL'])

        return loss

    def backward(self, loss, engine_state=None):
        optimizer = self.optimizers['unet_adam']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def prepare_output(self, preds_tuple, loss, batch, epoch=None, convert_one_hot=False):

        raw_preds, preds = preds_tuple

        # preds.requires_grad = False
        preds = preds.detach()
        preds.requires_grad = False

        output = {
            'loss': loss.item(),
        }
        if self.HP.metrics['detection']:
            if self.HP.detect_sizes or self.HP.detection['per_size']:
                output['label_tuple'] = (batch['POINTS'].squeeze(1).detach(), batch['SIZES'].squeeze(1).detach())
            else:
                output['label_tuple'] = batch['POINTS'].squeeze(1).detach()
            preds[~batch['MASK'].bool()] = 0.0
            output['preds_detection'] = preds.squeeze(1).detach()
        return output

    def create_smoothed_labels(self, batch):
        smoothed_labels = np.zeros(shape=(len(batch['POINTS']), self.HP.size[0], self.HP.size[1], self.HP.size[2]))
        point_labels = np.zeros(shape=(len(batch['POINTS']), self.HP.size[0], self.HP.size[1], self.HP.size[2]))
        for i, points in enumerate(batch['POINTS']):
            points = points[0]
            if torch.sum(points) != 0:
                point_labels[i, points[0], points[1], points[2]] = 1.0
                point_labels[i, 0, 0, 0] = 0.0
                smoothed_labels[i] = ndimage.gaussian_filter(point_labels[i], sigma=self.HP.gaussian_filter['sigma'])
        batch['LABEL'] = torch.FloatTensor(smoothed_labels).unsqueeze(1)
        batch['POINT_LABEL'] = torch.FloatTensor(point_labels).unsqueeze(1)
        return batch

    def train_batching(self, batch: Union[Dict, Tensor]) -> Union[Dict, List, Tensor, Tuple]:
        # create smoothed labels
        batch = self.create_smoothed_labels(batch)

        # Move to GPU
        for key, value in batch.items():
            batch[key] = value.to(self.devices[0], non_blocking=True)

        # Apply transforms
        batch['MRI'] = CustomTransforms.denoise(batch['MRI'], batch['MASK'])
        batch['MRI'] = CustomTransforms.standardize(batch['MRI'], batch['MASK'])

        if self.bin is not None:
            batch['COUNT'] = self.bin(batch['RAW_COUNT']).to(self.devices[0])

        # Augment
        for augmentation in self.augmentations:
            batch = augmentation(batch)

        return batch

    def evaluate_batching(self, batch: Union[Dict, Tensor]) -> Union[Dict, List, Tensor, Tuple]:
        # create smoothed labels
        batch = self.create_smoothed_labels(batch)

        # Move to GPU
        for key, value in batch.items():
            batch[key] = value.to(self.devices[0], non_blocking=True)

        # Apply transforms
        batch['MRI'] = CustomTransforms.denoise(batch['MRI'], batch['MASK'])
        batch['MRI'] = CustomTransforms.standardize(batch['MRI'], batch['MASK'])

        if self.bin is not None:
            batch['COUNT'] = self.bin(batch['RAW_COUNT']).to(self.devices[0])

        if self.HP.resize:
            batch = CustomTransforms.resize(batch, size=(64, 192, 192), imaging_keys=list(batch.keys()))

        return batch

    def train_step(self, engine: Engine, batch: Union[Dict, Tensor]) -> Union[Dict, Tensor, None]:
        batch = self.train_batching(batch)

        preds = self.forward(batch, mode='train', epoch=engine.state.epoch)

        loss = self.calculate_losses(preds, batch, epoch=engine.state.epoch)

        self.backward(loss, engine_state=engine.state)

        return self.prepare_output(preds, loss, batch, epoch=engine.state.epoch)

    @torch.no_grad()
    def evaluate_step(self, engine: Engine, batch: Union[Dict, Tensor]) -> Union[Dict, Tensor, None]:
        batch = self.evaluate_batching(batch)

        preds = self.forward(batch, mode='eval', epoch=engine.state.epoch)

        loss = self.calculate_losses(preds, batch, epoch=engine.state.epoch)

        return self.prepare_output(preds, loss, batch, epoch=engine.state.epoch)

    @torch.no_grad()
    def test_step(self, engine: Engine, batch: Union[Dict, Tensor]) -> Union[Dict, Tensor, None]:
        batch = self.evaluate_batching(batch)

        preds = self.forward(batch, mode='eval', epoch=engine.state.epoch)

        loss = self.calculate_losses(preds, batch, epoch=engine.state.epoch)

        return self.prepare_output(preds, loss, batch, epoch=engine.state.epoch)

    def setup_train_metrics(self) -> None:
        assembler = MetricAssembler()

        # Average loss
        avg_loss_metric = Average(output_transform=lambda output: output['loss'], device=self.devices[0])
        assembler.add(avg_loss_metric, 'train_avg_loss')

        self.train_metrics = assembler.build()

    def setup_evaluate_metrics(self) -> None:
        assembler = MetricAssembler()

        # Average loss
        avg_loss_metric = Average(output_transform=lambda output: output['loss'], device=self.devices[0])
        assembler.add(avg_loss_metric, 'val_avg_loss')

        if self.HP.metrics['detection']:
            # dcm F1
            dcm = DetectionConfusionMatrix(pred_key='preds_detection', label_key='label_tuple', **self.HP.detection)
            F1_detection = dcmF1(dcm)
            assembler.add(F1_detection, 'val_F1_detection')

        self.evaluate_metrics = assembler.build()

    def setup_test_metrics(self) -> None:
        assembler = MetricAssembler()

        # Average loss
        avg_loss_metric = Average(output_transform=lambda output: output['loss'], device=self.devices[0])
        assembler.add(avg_loss_metric, 'test_avg_loss')

        if self.HP.metrics['detection']:
            # dcm F1
            dcm = DetectionConfusionMatrix(pred_key='preds_detection', label_key='label_tuple', **self.HP.detection)
            F1_detection = dcmF1(dcm)
            assembler.add(F1_detection, 'test_F1_detection')

        self.test_metrics = assembler.build()


class ExperimentSegmentationTemplate(ExperimentTemplate):
    def __init__(self, Hyperparameters):
        super(ExperimentSegmentationTemplate, self).__init__(Hyperparameters)

    def setup_optimizers(self)->None:
        self.optimizers['unet_adam'] = Adam(self.models['unet'].parameters(), **self.HP.optimizers['unet_adam'])

    def setup_models(self) -> None:
        unet = NNUNet(**self.HP.models['unet'])
        unet = unet.to(self.devices[0])
        self.models['unet'] = unet

    def prepare_output(self, preds, loss, batch, eval=False):
        preds = preds.detach()
        preds.requires_grad = False
        output = super().prepare_output(preds, loss, batch)
        if eval:
            if self.HP.detect_sizes or self.HP.detection['per_size']:
                output['label_tuple'] = (batch['POINTS'].squeeze(1).detach(), batch['SIZES'].squeeze(1).detach())
            else:
                output['label_tuple'] = batch['POINTS'].squeeze(1).detach()
            preds_detection = torch.sigmoid(preds)
            preds_detection[~batch['MASK'].bool()] = 0.0
            output['preds_detection'] = preds_detection.squeeze(1).detach()
        return output

    @torch.no_grad()
    def evaluate_step(self, engine: Engine, batch: Union[Dict, Tensor]) -> Union[Dict, Tensor, None]:
        batch = self.evaluate_batching(batch)

        preds = self.forward(batch, mode='eval')

        loss = self.calculate_losses(preds, batch, engine.state.epoch)

        return self.prepare_output(preds, loss, batch, eval=True)

    @torch.no_grad()
    def test_step(self, engine: Engine, batch: Union[Dict, Tensor]) -> Union[Dict, Tensor, None]:
        batch = self.evaluate_batching(batch)

        preds = self.forward(batch, mode='eval')

        loss = self.calculate_losses(preds, batch, engine.state.epoch)

        return self.prepare_output(preds, loss, batch, eval=True)

    @torch.no_grad()
    def evaluate_test_step(self, engine: Engine, batch: Union[Dict, Tensor]) -> Union[Dict, Tensor, None]:
        batch = self.evaluate_batching(batch)

        preds = self.forward(batch, mode='eval')

        loss = self.calculate_losses(preds, batch, engine.state.epoch)

        output = self.prepare_output(preds, loss, batch)
        return output

    def setup_evaluate_metrics(self) -> None:
        assembler = MetricAssembler()

        # Average loss
        avg_loss_metric = Average(output_transform=lambda output: output['loss'], device=self.devices[0])
        assembler.add(avg_loss_metric, 'val_avg_loss')

        # btcm PR AUC, ROC AUC, F1
        btcm = BinaryThresholdConfusionMatrix(thresholds=torch.tensor(self.HP.metric_thresholds), pred_key='preds',
                                              label_key='label', device=self.devices[0])
        PR_AUC = ApproximateMetrics.ApproxPR_AUC(btcm)
        ROC_AUC = ApproximateMetrics.ApproxROC_AUC(btcm)
        F1_reduce = btcmF1(btcm)
        F1 = btcmF1(btcm, reduce=False)
        assembler.add(PR_AUC, 'val_PR_AUC')
        assembler.add(ROC_AUC, 'val_ROC_AUC')
        assembler.add(F1, 'val_F1_all')
        assembler.add(F1_reduce, 'val_F1')

        if self.HP.metrics['detection']:
            # dcm F1, cost
            dcm = DetectionConfusionMatrix(pred_key='preds_detection', label_key='label_tuple', **self.HP.detection)
            F1_detection = dcmF1(dcm)
            assembler.add(F1_detection, 'val_F1_detection')

        self.evaluate_metrics = assembler.build()

    def setup_test_metrics(self) -> None:
        assembler = MetricAssembler()

        # Average loss
        avg_loss_metric = Average(output_transform=lambda output: output['loss'], device=self.devices[0])
        assembler.add(avg_loss_metric, 'test_avg_loss')

        # btcm PR AUC, ROC AUC, F1
        btcm = BinaryThresholdConfusionMatrix(thresholds=torch.tensor(self.HP.metric_thresholds), pred_key='preds',
                                              label_key='label', device=self.devices[0])
        PR_AUC = ApproximateMetrics.ApproxPR_AUC(btcm)
        ROC_AUC = ApproximateMetrics.ApproxROC_AUC(btcm)
        F1_reduce = btcmF1(btcm)
        F1 = btcmF1(btcm, reduce=False)
        assembler.add(PR_AUC, 'test_PR_AUC')
        assembler.add(ROC_AUC, 'test_ROC_AUC')
        assembler.add(F1, 'test_F1_all')
        assembler.add(F1_reduce, 'test_F1')

        if self.HP.metrics['detection']:
            # dcm F1, cost
            dcm = DetectionConfusionMatrix(pred_key='preds_detection', label_key='label_tuple', **self.HP.detection)
            F1_detection = dcmF1(dcm)
            assembler.add(F1_detection, 'test_F1_detection')

        self.test_metrics = assembler.build()


