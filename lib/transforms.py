import numpy as np
import mxnet as mx
from mxnet.gluon.nn import Block, HybridBlock


class AlbumentationTransform(Block):

    def __init__(self, augmentation):
        super(AlbumentationTransform, self).__init__()
        self.augmentation = augmentation

    def forward(self, x):
        return mx.nd.array(self.augmentation(image=x.asnumpy())['image'])


class MixUp(Block):

    def __init__(self, params, prob):
        super(MixUp, self).__init__()
        self.mix_params = params
        self.prob = prob
        self.prev_image = (None, None)

    def forward(self, x, labels):
        pass