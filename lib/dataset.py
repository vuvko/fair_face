import numpy as np
import mxnet as mx
from mxnet import io
from mxnet import recordio
import pandas as pd
import numbers
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
        if self.augs is not None:
            img = self.augs(image=img)['image']
        img = img.transpose((2, 0, 1)).astype(np.float32)
        label = self.label_map[self.data[idx][1]]
        return mx.nd.array(img), label


class ImgRecDataset(Dataset):

    def __init__(self, rec_path: Path, augs=None):
        self.rec_path = rec_path
        path_imgidx = rec_path.with_suffix('.idx')
        self.augs = augs
        self.imgrec = recordio.MXIndexedRecordIO(str(path_imgidx), str(rec_path), 'r')
        s = self.imgrec.read_idx(0)
        header, _ = recordio.unpack(s)
        if header.flag > 0:
            self.header0 = (int(header.label[0]), int(header.label[1]))
            self.imgidx = []
            self.id2range = {}
            self.seq_identity = range(int(header.label[0]), int(header.label[1]))
            for identity in self.seq_identity:
                s = self.imgrec.read_idx(identity)
                header, _ = recordio.unpack(s)
                a, b = int(header.label[0]), int(header.label[1])
                self.id2range[identity] = (a, b)
                self.imgidx += range(a, b)
        else:
            self.imgidx = list(self.imgrec.keys)
        self.seq = self.imgidx
        prop_path = rec_path.parent / 'property'
        with open(str(prop_path), 'r') as f:
            self.num_labels = int(f.readline().split(',')[0].strip())

    def __len__(self):
        return len(self.seq)

    def __getitem__(self, idx):
        idx = self.seq[idx]
        s = self.imgrec.read_idx(idx)
        header, img = recordio.unpack(s)
        img = mx.image.imdecode(img)
        if self.augs is not None:
            img = mx.nd.array(self.augs(image=img.asnumpy())['image'])
        img = img.transpose((2, 0, 1)).astype(np.float32)
        label = header.label
        if not isinstance(label, numbers.Number):
            label = label[0]
        return img, label


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
