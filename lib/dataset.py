import numpy as np
import mxnet as mx
import pandas as pd
from pathlib import Path
from more_itertools import unzip
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.data import Dataset
from .data import load_img
from .mytypes import MxImg, ItemFilter
from .filter import true_filter, ExistsFilter
from typing import Sequence, Tuple, Optional, Iterable


class ImgDataset(Dataset):

    def __init__(self, paths: Sequence[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> MxImg:
        return mx.img.imread(str(self.paths[idx])).transpose((2, 0, 1)).astype(np.float32)


class InfoDataset(Dataset):

    def __init__(self, data_info: pd.DataFrame, augs=None, filter_fn: ItemFilter = true_filter):
        self.data_info = data_info
        self.data = list(filter(filter_fn, zip(data_info['img_path'], data_info['SUBJECT_ID'])))
        self.num_labels = None
        self.label_map = {}
        self.remap_labels()
        self.augs = augs

    def remap_labels(self):
        num_labels = 0
        for _, cur_label in self.data:
            if cur_label not in self.label_map:
                self.label_map[cur_label] = num_labels
                num_labels += 1
        self.num_labels = num_labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[MxImg, int]:
        img = load_img(self.data[idx][0])
        if self.augs:
            img = self.augs(image=img)['image']
        img = img.transpose((2, 0, 1)).astype(np.float32)
        label = self.label_map[self.data[idx][1]]
        return mx.nd.array(img), label


class PairDataset(Dataset):

    def __init__(self, path_pairs: Iterable[Tuple[Path, Path]], labels: Iterable[int], augs=None):
        filter_fn = ExistsFilter()
        self.data = list(filter(lambda item: filter_fn((item[0][0], 0)) and filter_fn((item[0][1], 0)),
                                zip(path_pairs, labels)))
        self.augs = augs

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[MxImg, MxImg, int]:
        paths, label = self.data[idx]
        left_img = load_img(paths[0])
        right_img = load_img(paths[1])
        if self.augs:
            left_img = self.augs(image=left_img)['image']
            right_img = self.augs(image=right_img)['image']
        left_img = left_img.transpose((2, 0, 1)).astype(np.float32)
        right_img = right_img.transpose((2, 0, 1)).astype(np.float32)
        return mx.nd.array(left_img), mx.nd.array(right_img), label
