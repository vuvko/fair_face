import numpy as np
import mxnet as mx
import sys
from mxnet.gluon.data import Dataset
from pathlib import Path

from ..mytypes import MxImg
from typing import Sequence, Optional, Tuple


class PrepareDataset(Dataset):

    def __init__(self, img_list: Sequence[Path], root: Optional[Path] = None):
        super(PrepareDataset, self).__init__()
        self.img_list = img_list
        self.root = root

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, idx: int) -> Tuple[MxImg, int, int, int]:
        if self.root is not None:
            img_path = self.root / self.img_list[idx]
        else:
            img_path = self.img_list[idx]
        try:
            img = mx.img.imread(str(img_path))
        except:
            e = sys.exc_info()[0]
            print('Error on loading img', img_path, ':', e)
            img = mx.nd.zeros((1, 1, 3), dtype=np.uint8)
        height, width = img.shape[:2]
        return img, idx, width, height
