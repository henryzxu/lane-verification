import cv2

import numpy as np
import torch
from torchvision.transforms import Normalize as Normalize_th
import matplotlib.pyplot as plt


class CustomTransform:
    def __call__(self, *args, **kwargs):
        raise NotImplementedError

    def __str__(self):
        return self.__class__.__name__

    def __eq__(self, name):
        return str(self) == name

    def __iter__(self):
        def iter_fn():
            for t in [self]:
                yield t
        return iter_fn()

    def __contains__(self, name):
        for t in self.__iter__():
            if isinstance(t, Compose):
                if name in t:
                    return True
            elif name == t:
                return True
        return False


class Compose(CustomTransform):
    """
    All transform in Compose should be able to accept two non None variable, img and boxes
    """
    def __init__(self, *transforms):
        self.transforms = [*transforms]

    def __call__(self, sample):
        for t in self.transforms:
            sample = t(sample)
        return sample

    def __iter__(self):
        return iter(self.transforms)

    def modules(self):
        yield self
        for t in self.transforms:
            if isinstance(t, Compose):
                for _t in t.modules():
                    yield _t
            else:
                yield t


class Resize(CustomTransform):
    def __init__(self, size, interpolation=cv2.INTER_AREA):
        if isinstance(size, int):
            size = (size, size)
        self.size = size  #(W, H)
        self.interpolation = interpolation

    def __call__(self, sample):
        img = sample.get('img')
        segLabel = sample.get('segLabel', None)

        img = cv2.resize(img, self.size, interpolation=self.interpolation)
        if segLabel is not None:
            segLabel = cv2.resize(segLabel, self.size, interpolation=cv2.INTER_NEAREST)

        _sample = sample.copy()
        _sample['img'] = img
        _sample['segLabel'] = segLabel
        return _sample

    def reset_size(self, size):
        if isinstance(size, int):
            size = (size, size)
        self.size = size



class ExtractRandomPatch(CustomTransform):
    def __init__(self, patch_height, min_patch, interpolation=cv2.INTER_AREA):
        self.patch_height = patch_height
        self.min_patch = min_patch
        self.interpolation = interpolation

    def __call__(self, sample):
        img = sample.get('img')
        segLabel = sample.get('segLabel', None)
        # print(sub_patch_idx)

        min_height = self.patch_height * self.min_patch
        if img.shape[0] < min_height:
            img = cv2.resize(img, (img.shape[1], min_height), interpolation=self.interpolation)

        rand_idx = np.random.randint(self.patch_height, img.shape[0])

        img = img[rand_idx - self.patch_height: rand_idx]



        _sample = sample.copy()
        _sample['img'] = img
        _sample['segLabel'] = segLabel
        return _sample



class ExtractPatch(CustomTransform):
    def __init__(self, patch_height, min_patch, patch_offset, interpolation=cv2.INTER_AREA):
        self.patch_height = patch_height
        self.min_patch = min_patch
        self.patch_offset = patch_offset
        self.interpolation = interpolation

    def __call__(self, sample):
        img = sample.get('img')
        segLabel = sample.get('segLabel', None)

        patch_idx = sample.get('patch_idx')
        main_patch_idx = patch_idx // 3
        sub_patch_idx = (patch_idx % 3) * self.patch_height // 3
        # print(sub_patch_idx)

        min_height = self.patch_height * self.min_patch
        if img.shape[0] < min_height:
            img = cv2.resize(img, (img.shape[1], min_height), interpolation=self.interpolation)

        img = img[img.shape[0] - self.patch_offset * patch_idx - self.patch_height: img.shape[0] - self.patch_offset * patch_idx]



        _sample = sample.copy()
        _sample['img'] = img
        _sample['segLabel'] = segLabel
        return _sample



class RandomResize(Resize):
    """
    Resize to (w, h), where w randomly samples from (minW, maxW) and h randomly samples from (minH, maxH)
    """
    def __init__(self, minW, maxW, minH=None, maxH=None, batch=False):
        if minH is None or maxH is None:
            minH, maxH = minW, maxW
        super(RandomResize, self).__init__((minW, minH))
        self.minW = minW
        self.maxW = maxW
        self.minH = minH
        self.maxH = maxH
        self.batch = batch

    def random_set_size(self):
        w = np.random.randint(self.minW, self.maxW+1)
        h = np.random.randint(self.minH, self.maxH+1)
        self.reset_size((w, h))


class Rotation(CustomTransform):
    def __init__(self, theta):
        self.theta = theta

    def __call__(self, sample):
        img = sample.get('img')
        segLabel = sample.get('segLabel', None)

        u = np.random.uniform()
        degree = (u-0.5) * self.theta
        R = cv2.getRotationMatrix2D((img.shape[1]//2, img.shape[0]//2), degree, 1)
        img = cv2.warpAffine(img, R, (img.shape[1], img.shape[0]), flags=cv2.INTER_LINEAR)
        if segLabel is not None:
            segLabel = cv2.warpAffine(segLabel, R, (segLabel.shape[1], segLabel.shape[0]), flags=cv2.INTER_NEAREST)

        _sample = sample.copy()
        _sample['img'] = img
        _sample['segLabel'] = segLabel
        return _sample

    def reset_theta(self, theta):
        self.theta = theta


class Normalize(CustomTransform):
    def __init__(self, mean, std):
        self.transform = Normalize_th(mean, std)

    def __call__(self, sample):
        img = sample.get('img')

        img = self.transform(img)

        _sample = sample.copy()
        _sample['img'] = img
        return _sample


class ToTensor(CustomTransform):
    def __init__(self, dtype=torch.float):
        self.dtype=dtype

    def __call__(self, sample):
        img = sample.get('img')
        segLabel = sample.get('segLabel', None)

        orig_img = img

        img = img.transpose(2, 0, 1)
        img = torch.from_numpy(img).type(self.dtype)
        if np.count_nonzero(orig_img>1):
            img = img/255.
        if segLabel is not None:
            segLabel = torch.from_numpy(segLabel).type(torch.long)

        _sample = sample.copy()
        _sample['img'] = img
        _sample['segLabel'] = segLabel
        return _sample


class Flatten(CustomTransform):

    def __call__(self, sample):
        img = sample.get('img')
        segLabel = sample.get('segLabel', None)

        img = np.ravel(img.detach().cpu().numpy())
        if segLabel:
            segLabel = segLabel.detach().cpu().numpy()


        _sample = sample.copy()
        _sample['img'] = img
        _sample['segLabel'] = segLabel
        return _sample


