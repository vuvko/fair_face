import mxnet as mx

from typing import Optional


class EpochMultiplier(mx.gluon.data.sampler.Sampler):

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
