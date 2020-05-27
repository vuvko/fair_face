import numpy as np
import mxnet as mx
from pathlib import Path
from lib.compare import CompareModel
from lib.submit import submit


if __name__ == '__main__':
    comparator = CompareModel(ctx=mx.gpu(0))
    val_dir = Path('/run/media/andrey/Fast/FairFace/data_prep2/val/data')
    val_csv = Path('/run/media/andrey/Fast/FairFace/data/val/evaluation_pairs/predictions.csv')
    submit_name = 'test_arcface_allval'
    submit(val_dir, val_csv, comparator, submit_name)
