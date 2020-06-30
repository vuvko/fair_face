import numpy as np
import mxnet as mx
from pathlib import Path
from lib.compare import CompareModel
from lib.submit import submit


if __name__ == '__main__':
    comparator = CompareModel('/home/andrey/.insightface/models/VGG2-ResNet50-Arcface/model', 0, ctx=mx.gpu(0))
    val_dir = Path('/run/media/andrey/Fast/FairFace/test_wide/data')
    val_csv = Path('/run/media/andrey/Fast/FairFace/test/evaluation_pairs/predictions.csv')
    submit_name = 'final_arcface_vgg_flip_allval'
    submit(val_dir, val_csv, comparator, submit_name)
