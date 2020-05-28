import numpy as np
import mxnet as mx
from mxnet.gluon.nn import Block, HybridBlock


class CrossEntropyLoss(HybridBlock):

    def __init__(self):
        super(CrossEntropyLoss, self).__init__()

    def hybrid_forward(self, F, preds, labels):
        return -F.sum(labels * F.log_softmax(preds))
