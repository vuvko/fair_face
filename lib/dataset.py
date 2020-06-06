import numpy as np
import mxnet as mx
import pandas as pd
from pathlib import Path
from more_itertools import unzip
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.data import Dataset
from .data import load_img
from .mytypes import MxImg, ItemFilter
from .filter import true_filter
from typing import Sequence, Tuple, Optional


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
        self.num_labels = len(np.unique(list(unzip(self.data)[1])))
        self.augs = augs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int) -> Tuple[MxImg, int]:
        img = load_img(self.data[idx][0])
        if self.augs:
            img = self.augs(image=img)['image']
        img = img.transpose((2, 0, 1)).astype(np.float32)
        label = self.data[idx][1]
        return mx.nd.array(img), label
