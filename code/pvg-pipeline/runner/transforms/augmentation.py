from typing import Union, Tuple, Optional
from torch import Tensor, tensor
import torch
import torch.nn.functional as F
import numpy as np
from kornia.augmentation.random_generator import random_affine_generator3d
from kornia.augmentation.functional.functional3d import apply_affine3d
from kornia.augmentation.augmentation3d import RandomHorizontalFlip3D, RandomVerticalFlip3D, RandomDepthicalFlip3D


class CustomTransforms:
    '''
        Custom transform methods
        implements randdom erase and wrapped dict implementations
        implements wrapped random affine based class and extensions
    '''
    @staticmethod
    @torch.no_grad()
    def denoise(images: Tensor, masks: Tensor)->Tensor:
        '''
            Skull stripping with mask
            @param: images: Tensor; batches of images with shape [batch,channel,Z,Y,X]
            @param: masks: Tensor; brain mask of shape [batch, 1 ,Z,Y,X]
            @return Tensor ~ transformed images
        '''
        images = images.mul(masks)
        return images

    @staticmethod
    @torch.no_grad()
    def standardize(images: Tensor, masks: Tensor, eps: float=1e-12)->Tensor:
        '''
            Standardize to 0 mean, 1 std. dev. in brain (masked/skull stripped)
            @param: images: Tensor; batches of images with shape [batch,channel,Z,Y,X]
            @param: masks: Tensor; brain mask of shape [batch, 1 ,Z,Y,X]
            @return Tensor ~ transformed image
        '''
        images = images.mul(masks)
        N = masks.sum(dim=(2,3,4), keepdims=True)
        means = images.sum(dim=(2,3,4), keepdims=True) / N
        stds = torch.sqrt(images.pow(2).sum(dim=(2,3,4), keepdims=True) / N - means.pow(2) + eps)
        images = images.sub(means)
        images = images.div(stds)
        images = images.mul(masks)
        return images

    @staticmethod
    @torch.no_grad()
    def resize_images(images: Tensor, size: Tuple)->Tensor:
        '''
            Resizing to standard size
            @param: images: Tensor; batches of images with shape [batch,channel,Z,Y,X]
            @return Tensor ~ resized images
        '''
        images_ = torch.zeros(size=(1, images.size(1), size(0), size(1), size(2)), device=images.device)
        for i in range(len(images)):
            img = images[i]
            img_size = img.size()[-3:]
            size_diff = np.subtract(size, img_size)
            if not np.all(size_diff >= 0):
                # print(f"Warning image size larger than requested size: {img_size}")
                size_tmp = np.subtract(size, img_size)
                size_tmp = -1*np.where(size_tmp > 0, 0, size_tmp)
                indices = np.zeros(shape=6)
                indices[[0, 2, 4]] = size_tmp // 2
                indices[[1, 3, 5]] = img_size - (size_tmp - indices[[0, 2, 4]])
                indices = indices.astype(np.int32)
                # print("indices:", indices)
                img = img[:, indices[0]:indices[1], indices[2]:indices[3], indices[4]:indices[5]]
                ind = np.argwhere(size_diff < 0)
                size_diff[ind] = 0
            padding = np.zeros(shape=(8))
            padding[[2, 4, 6]] = size_diff // 2
            padding[[3, 5, 7]] = size_diff - padding[[2, 4, 6]]
            padding = np.flip(padding.astype(np.int32))

            images_[i] = F.pad(input=img, pad=tuple(padding), mode='constant', value=0)
        del images
        return images_

    @staticmethod
    @torch.no_grad()
    def resize(batch, size: Tuple, imaging_keys: list)->Tensor:
        '''
            Resizing to standard size
            @param: images: Tensor; batches of images with shape [batch,channel,Z,Y,X]
            @param: masks: Tensor; brain mask of shape [batch, 1 ,Z,Y,X]
            @return Tensor ~ resized images
        '''
        for key in batch.keys():
            if key not in imaging_keys:
                continue
            batch[key] = CustomTransforms.resize_images(batch[key], size)

        return batch


    @staticmethod
    @torch.no_grad()
    def bind(images: Tensor, masks: Tensor, eps: float=1e-12)->Tensor:

        '''
            Normalizing to 0 min, 1 max, in brain (masked/skull stripped), 3 std. dev. cutoff
            @param: images: Tensor; batches of images with shape [batch,channel,Z,Y,X]
            @param: masks: Tensor; brain mask of shape [batch, 1 ,Z,Y,X]
        '''

        images = images.mul(masks)
        b,c,_,_,_ = images.size()
        N = masks.sum(dim=(2,3,4), keepdims=True) # across Z,Y,X dimensions
        means = images.sum(dim=(2,3,4), keepdims=True) / N
        stds = torch.sqrt(images.pow(2).sum(dim=(2,3,4), keepdims=True) / N - means.pow(2) + eps)
        images = images.mul(masks) # background clean up
        images = torch.maximum(images, means-3*stds)
        images = torch.minimum(images, means+3*stds)
        images = images.add( (1-masks)*1e10) # set background to high number to get only min. of brain
        images = images.sub(torch.min(images.view(b,c,-1), dim=2)[0].view(b,c,1,1,1))
        images = images.mul(masks) # background clean up
        images = images.div(torch.max(images.view(b,c,-1), dim=2)[0].view(b,c,1,1,1))
        images = images.mul(masks) # background clean up

        return images

    @staticmethod
    @torch.no_grad()
    def std_bind(images: Tensor, masks: Tensor)->Tensor:

        '''
            On a standardized image: normalize to 0 min, 1 max, in brain (masked/skull stripped), 3 std. dev. cutoff
            @param: images: Tensor; batches of images with shape [batch,channel,Z,Y,X]
            @param: masks: Tensor; brain mask of shape [batch, 1 ,Z,Y,X]
            @return Tensor ~ transformed images
        '''

        masks = masks.squeeze()
        images = images.transpose(1,0)
        c, b, _, _, _ = images.size()
        images = torch.clip(images, -3, 3)
        images = images.sub(torch.min(images.view(c,b,-1), dim=2)[0].view(c,b,1,1,1))
        images = images.div(torch.max(images.view(c,b,-1), dim=2)[0].view(c,b,1,1,1))
        images = images.transpose(1,0)
        images = images.mul(masks.unsqueeze(1)) # denoise/mask background

        return images

    class _AffineSampler3D():

        ''' base class for any affine transformations '''

        def __init__(self,
                    degrees: Optional[
                                Union[Tuple[float, float],
                                      Tuple[
                                            Tuple[float, float],
                                            Tuple[float, float],
                                            Tuple[float, float]]]] = None,
                    translate: Optional[Tuple[float, float, float]] = None,
                    scale: Union[Tuple[float, float, float],
                                 Tuple[
                                    Tuple[float, float, float],
                                    Tuple[float, float, float],
                                    Tuple[float, float, float]]] = None,
                    shear: Optional[
                                Union[Tuple[float, float],
                                      Tuple[
                                            Tuple[float, float],
                                            Tuple[float, float],
                                            Tuple[float, float],
                                            Tuple[float, float],
                                            Tuple[float, float],
                                            Tuple[float, float]]]] = None,
                    same_on_batch: bool = False,
                    device = torch.device('cpu')):

            self.degrees = tensor(degrees, device=device, dtype=torch.float) if degrees is not None else torch.zeros(3,2, device=device)
            self.translate = tensor(translate, device=device, dtype=torch.float) if translate is not None else torch.zeros(3, device=device)
            self.scale = tensor(scale, device=device, dtype=torch.float) if scale is not None else torch.ones(3,2, device=device)
            self.shear = tensor(shear, device=device, dtype=torch.float) if shear is not None else torch.ones(6, 2,
                                                                                                              device=device)
            self.same_on_batch = same_on_batch
            self.device = device

            # If passed tuple of shape (2), expand to (3,2) to fit to kornia
            if self.scale.shape != torch.Size([3, 2]):
                if self.scale.shape == torch.Size([2]):
                    self.scale = self.scale.unsqueeze(0).repeat(3, 1)
                else:
                    raise Exception("Scale needs to be in proper shape of (3,2)!")

            # If passed tuple of shape (2), expand to (3,2) to fit to kornia
            if self.degrees.shape != torch.Size([3, 2]):
                if self.degrees.shape == torch.Size([2]):
                    self.degrees = self.degrees.unsqueeze(0).repeat(3, 1)
                else:
                    raise Exception("Degrees needs to be in proper shape of (3,2)!")

            # If passed tuple of shape (2), expand to (6,2) to fit to kornia
            if self.shear.shape != torch.Size([6, 2]):
                if self.shear.shape == torch.Size([2]):
                    self.shear = self.shear.unsqueeze(0).repeat(6, 1)
                else:
                    raise Exception("Shear needs to be in proper shape of (6,2)!")

        def sample(self, batch_size: int, depth: int, height: int, width: int):
            return random_affine_generator3d(batch_size, depth, height, width, self.degrees, self.translate, self.scale,
                                             self.shear, self.same_on_batch, self.device)

    class WrappedRandomAffine3D(_AffineSampler3D):
        '''
            implements an affine transformation over a batch of images
            wrappered for pipeline integration
            degrees: float ranges
            translate: float float float ~ factors
            scale: float ranges
            shear: (6,2)
            input mapping:
                - dict[imaging_keys, Tuple(resample flag: int, corners flag: bool)]
            resample flag:
                0 -> NN resmaple
                1 -> Trilinear reample
            corerns flag:
                - True -> align corners
                - False ->  dont align corners
        '''

        def __init__(self,
                     input_mapping: dict,
                     degrees: Optional[Tensor] = None,
                     translate: Optional[Tensor] = None,
                     scale: Optional[Tensor] = None,
                     shear: Optional[Tensor] = None,
                     p: float = 1.0,
                     same_on_batch: bool = False,
                     device=torch.device('cpu')):

            self.mapping = {k: {
                'resample': tensor(i, dtype=torch.float),
                'align_corners': tensor(j, dtype=torch.bool)
                    } for k, (i, j) in input_mapping.items()}
            self.p = p
            super().__init__(degrees, translate, scale, shear, same_on_batch, device)

        @torch.no_grad()
        def __call__(self, info):
            if torch.rand(1) > self.p: return info
            b, _, d, h, w = info[list(self.mapping.keys())[0]].size()
            affine_params = self.sample(b, d, h, w)
            for imaging_key, resample_args in self.mapping.items():
                info[imaging_key] = apply_affine3d(info[imaging_key], affine_params, resample_args)
            return info

    class WrappedRandomHorizontalFlip():
        '''
            implements a horizontal flip with probability p
            input mapping: List of imaging keys
            p: probability of flipping
        '''
        def __init__(self,
                    input_mapping: list,
                    p: float = 1.0,
                    device = torch.device('cpu')):

            self.mapping = input_mapping
            self.p = p
            self.flip = RandomHorizontalFlip3D(p=1.0)
            self.device = device

        @torch.no_grad()
        def __call__(self, info):
            sample = np.random.random()
            if sample < self.p:
                for key in self.mapping:
                    imgs = info[key].clone()
                    info[key] = self.flip(imgs)
            return info

    class WrappedRandomVerticalFlip():
        '''
            implements a vertical flip with probability p
            input mapping: List of imaging keys
            p: probability of flipping
        '''
        def __init__(self,
                    input_mapping: list,
                    p: float = 1.0,
                    device = torch.device('cpu')):

            self.mapping = input_mapping
            self.p = p
            self.flip = RandomVerticalFlip3D(p=1.0)
            self.device = device

        @torch.no_grad()
        def __call__(self, info):
            sample = np.random.random()
            if sample < self.p:
                for key in self.mapping:
                    imgs = info[key].clone()
                    info[key] = self.flip(imgs)
            return info

    class WrappedRandomDepthicalFlip():
        '''
            implements a depthical flip with probability p
            input mapping: List of imaging keys
            p: probability of flipping
        '''
        def __init__(self,
                    input_mapping: list,
                    p: float = 1.0,
                    device = torch.device('cpu')):

            self.mapping = input_mapping
            self.p = p
            self.flip = RandomDepthicalFlip3D(p=1.0)
            self.device = device

        @torch.no_grad()
        def __call__(self, info):
            sample = np.random.random()
            if sample < self.p:
                for key in self.mapping:
                    imgs = info[key].clone()
                    info[key] = self.flip(imgs)
            return info

    class BinCount():
        def __init__(self, bins=[0, 1, 2, 3, 4]):
            '''
            Takes in a count value and returns which bin it belongs to
            If a count is in bins[i], it means it is strictly greater than bins[i-1] and less than or equal to bins[i]
            '''
            self.bins = bins

        def __call__(self, values):
            binned = torch.zeros(size=(len(values), len(self.bins)))
            for j, value in enumerate(values):
                one_hot = torch.zeros(len(self.bins))
                for i, bin in enumerate(self.bins):
                    if i == 0 and value <= bin:
                        one_hot[bin] = 1
                        break
                    if i != 0 and (self.bins[i - 1] < value <= bin):
                        one_hot[bin] = 1
                        break
                if torch.all(~one_hot.bool()):
                    one_hot[-1] = 1
                binned[j] = one_hot
            return binned