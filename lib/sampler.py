import mxnet as mx
import numpy as np
import random
from mxnet.gluon.data.sampler import Sampler

from typing import Optional


class EpochMultiplier(Sampler):

    def __init__(self,
                 dataset_len: int,
                 multiplier: int = 1,
                 other_sampler: Optional[mx.gluon.data.sampler.Sampler] = None):
        super(EpochMultiplier, self).__init__()
        self.dataset_len = dataset_len
        self.multiplier = multiplier
        if other_sampler is not None:
            self.dataset_len = len(other_sampler)
        self.other_sampler = other_sampler

    def __iter__(self):
        if self.other_sampler is None:
            seq = sum([list(range(self.dataset_len))] * self.multiplier, [])
        else:
            seq = sum([list(iter(self.other_sampler))] * self.multiplier, [])
        return iter(seq)

    def __len__(self) -> int:
        return self.dataset_len * self.multiplier


class UniformClassSampler(Sampler):

    def __init__(self, dataset: mx.gluon.data.Dataset, shuffle: bool = True):
        super(UniformClassSampler, self).__init__()
        self.num_classes = 0
        self.num_imgs = len(dataset)
        self.class_dict = dict()
        self.shuffle = shuffle
        for idx, (_, class_idx) in enumerate(dataset):
            if class_idx not in self.class_dict:
                self.class_dict[class_idx] = []
                self.num_classes += 1
            self.class_dict[class_idx].append(idx)

    def __iter__(self):
        avg_images = self.num_imgs // self.num_classes
        residual_images = self.num_imgs % self.num_classes
        seq = []
        for class_idx, class_sample in self.class_dict.items():
            if class_idx >= residual_images:
                residual = 0
            else:
                residual = 1
            seq.extend(np.random.choice(class_sample, avg_images + residual, replace=True))
        if self.shuffle:
            random.shuffle(seq)
        return iter(seq)

    def __len__(self) -> int:
        return self.num_imgs
