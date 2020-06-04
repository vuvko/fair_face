import numpy as np
import mxnet as mx
import pandas as pd
from pathlib import Path
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.data import Dataset
from .data import load_img
from .mytypes import MxImg
from typing import Sequence, Tuple


class ImgDataset(Dataset):

    def __init__(self, paths: Sequence[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> MxImg:
        return mx.img.imread(str(self.paths[idx])).transpose((2, 0, 1)).astype(np.float32)


class InfoDataset(Dataset):

    def __init__(self, data_info: pd.DataFrame):
        self.data_info = data_info
        self.paths = data_info['img_paths']
        self.labels = data_info['SUBJECT_ID']

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int) -> Tuple[MxImg, int]:
        img = load_img(self.paths[idx]).transpose((2, 0, 1))
        label = self.labels[idx]
        return mx.nd.array(img), label
