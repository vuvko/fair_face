from lib.validate import validate, plot_roc
from lib.compare import CompareModel, config_median_comparator, cluster_sklearn
from lib.data import load_info, sample_pairs, aggregate_subjects
from lib import metrics
from functools import partial
import mxnet as mx
import numpy as np
from tqdm import tqdm
from pathlib import Path
from more_itertools import unzip
from sklearn import cluster
from lib.submit import submit
import argparse
from typing import Sequence


val_csv = Path('/run/media/andrey/Fast/FairFace/data/val/evaluation_pairs/predictions.csv')
data_path = Path('/run/media/andrey/Fast/FairFace/fixed_data/val/data')
all_imgs = list(data_path.iterdir())
algorithm = cluster.AgglomerativeClustering(
        n_clusters=None,
        affinity='cosine',
        memory='/run/media/andrey/Data/cluster_cache/',
        linkage='complete',
        distance_threshold=0.92
    )
norm_median = False
median_alpha = 0.5
cur_exp = 'ultimate5'
model_path = Path('experiments') / cur_exp / 'snapshots'
cur_epoch = len(list(model_path.iterdir())) - 1
comparator = CompareModel(str(model_path / cur_exp), cur_epoch, use_flip=True, ctx=mx.gpu(0))
# metric = metrics.euclidean
# cluster_comparator_eu = config_median_comparator(comparator, partial(cluster_sklearn, algorithm=algorithm),
#                                                  val_data['img_path'], metric, norm_median, median_alpha)

metric = metrics.cosine
cluster_comparator_co = config_median_comparator(comparator, partial(cluster_sklearn, algorithm=algorithm),
                                                 all_imgs, metric, norm_median, median_alpha)

submit_name = 'ultimate5+cluster_cosine_f'
submit(data_path, val_csv, comparator, submit_name)
