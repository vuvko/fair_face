import numpy as np
import mxnet as mx
from mxnet.gluon.nn import Block, HybridBlock
from gluoncv.data.transforms.image import resize_long

from ..mytypes import Backend, BackendEl


class ResizeLong(Block):

    def __init__(self, size: int, interp: int = 2):
        super(ResizeLong, self).__init__()
        self.size = size
        self.interp = interp

    def forward(self, img: mx.nd.NDArray) -> mx.nd.NDArray:
        resized = resize_long(img, self.size, self.interp)
        height, width = resized.shape[:2]
        max_size = max(width, height)
        padded = mx.nd.empty((max_size, max_size, resized.shape[2]), ctx=resized.context, dtype=resized.dtype)
        padded[:height, :width] = resized
        if height > width:
            padded[:, width:] = 0
        elif height < width:
            padded[height:, :] = 0
        return padded


class NonNormalizedTensor(HybridBlock):

    def __init__(self):
        super(NonNormalizedTensor, self).__init__()

    def hybrid_forward(self, F: Backend, batch: BackendEl) -> BackendEl:
        return F.transpose(batch, (2, 0, 1))
