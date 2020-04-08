import numpy as np
import mxnet as mx
from pathlib import Path
from mxnet.gluon.data.vision import transforms
from mxnet.gluon.data import Dataset

from .mytypes import MxImg
from typing import Sequence, Tuple


class ImgDataset(Dataset):

    def __init__(self, paths: Sequence[Path]):
        self.paths = paths

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> MxImg:
        return mx.img.imread(str(self.paths[idx])).transpose((2, 0, 1)).astype(np.float32)
