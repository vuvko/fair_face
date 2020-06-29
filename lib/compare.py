import numpy as np
import mxnet as mx
from pathlib import Path
from insightface import model_zoo
from sklearn import preprocessing
from . import metrics
from .dataset import ImgDataset

from .mytypes import Embedding, Labels
from typing import List, Sequence, Optional


class CompareModel(object):

    def __init__(self,
                 model_name: str = 'arcface_r100_v1',
                 epoch_num: int = 0,
                 use_flip: bool = False,
                 ctx: mx.Context = mx.cpu()):
        self.use_flip = use_flip
        self.embeddings = dict()
        if model_name == 'arcface_r100_v1':
            model = model_zoo.get_model(model_name)
            if ctx.device_type.startswith('cpu'):
                ctx_id = -1
            else:
                ctx_id = ctx.device_id
            model.prepare(ctx_id=ctx_id)
            self.model = model.model
        else:
            sym, arg_params, aux_params = mx.model.load_checkpoint(model_name, epoch_num)
            sym = sym.get_internals()['fc1_output']
            model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
            data_shape = (2 if use_flip else 1, 3, 112, 112)
            model.bind(data_shapes=[('data', data_shape)], for_training=False)
            model.set_params(arg_params, aux_params)
            #warmup
            data = mx.nd.zeros(shape=data_shape)
            db = mx.io.DataBatch(data=(data,))
            model.forward(db, is_train=False)
            embedding = model.get_outputs()[0].asnumpy()
            self.model = model
        self.metric = metrics.cosine

    def get_embedding(self, im_path: Path) -> Embedding:
        if im_path not in self.embeddings:
            img = mx.img.imread(str(im_path))
            prep_img = img.transpose((2, 0, 1)).expand_dims(0).astype(np.float32)
            if self.use_flip:
                flipped = mx.nd.flip(img, axis=1).transpose((2, 0, 1)).expand_dims(0).astype(np.float32)
                img = mx.nd.concatenate([prep_img, flipped])
            else:
                img = prep_img
            batch = mx.io.DataBatch([img])
            self.model.forward(batch, is_train=False)
            self.embeddings[im_path] = self.model.get_outputs()[0].asnumpy().mean(axis=0)
        return self.embeddings[im_path]

    def prepare_all_embeddings(self,
                               paths: List[Path],
                               **loader_params) -> None:
        data = mx.gluon.data.DataLoader(ImgDataset(paths), shuffle=False, **loader_params)
        raise NotImplementedError

    def __call__(self, path1: Path, path2: Path) -> float:
        emb1 = self.get_embedding(path1)
        emb2 = self.get_embedding(path2)
        return self.metric(emb1, emb2)


def cluster_sklearn(embs, algorithm, dist_matrix=None):
    if dist_matrix is None:
        algorithm.fit(embs)
    else:
        algorithm.fit(dist_matrix)
    return algorithm.labels_


def get_medians(embs, labels, norm_median: bool = False):
    per_label = {}
    for cur_emb, cur_label in zip(embs, labels):
        if cur_label not in per_label:
            per_label[cur_label] = []
        per_label[cur_label].append(cur_emb)
    medias = {}
    for cur_label, cur_embs in per_label.items():
        if cur_label == -1:
            continue
        median = np.mean(cur_embs, axis=0)
        if norm_median:
            median = median / np.sqrt(np.sum(median ** 2, axis=-1, keepdims=True))
        medias[cur_label] = median
    ret_medians = np.empty(embs.shape, dtype=np.float32)
    for cur_idx, cur_label in enumerate(labels):
        if cur_label == -1:
            ret_medians[cur_idx] = embs[cur_idx]
        else:
            ret_medians[cur_idx] = medias[cur_label]
    return ret_medians


def config_median_comparator(comparator, label_method, all_paths: List[Path], metric, norm_median: bool = False,
                             median_alpha: float = 1.0):
    #     print('Preparing embeddings')
    res = [comparator(cur_p, cur_p) for cur_p in all_paths]
    embeddings_dict = comparator.embeddings
    embeddings = np.array([embeddings_dict[cur_p] for cur_p in all_paths])
    path_idx = {path: idx for idx, path in enumerate(all_paths)}
    #     print('Getting medians')
    labels = label_method(embeddings)
    medians = get_medians(embeddings, labels, norm_median)

    # embeddings = preprocessing.normalize(embeddings)
    # medians = preprocessing.normalize(medians)

    #     print('Done configurating')

    def compare(left_path: Path, right_path: Path):
        left_embedding = embeddings[path_idx[left_path]]
        left_median = medians[path_idx[left_path]]
        right_embedding = embeddings[path_idx[right_path]]
        right_median = medians[path_idx[right_path]]
        left_comp = left_median * median_alpha + left_embedding * (1 - median_alpha)
        right_comp = right_median * median_alpha + right_embedding * (1 - median_alpha)
        return metric(left_comp, right_comp)

    return compare


def config_rank_comparator(comparator, all_paths: List[Path], metric: str = 'cosine'):
    res = [comparator(cur_p, cur_p) for cur_p in all_paths]
    embeddings_dict = comparator.embeddings
    embeddings = np.array([embeddings_dict[cur_p] for cur_p in all_paths])
    path_idx = {path: idx for idx, path in enumerate(all_paths)}
    if metric == 'cosine':
        embeddings = embeddings / np.sqrt(np.sum(embeddings ** 2, axis=-1, keepdims=True))
    all_ranks = np.empty((embeddings.shape[0], embeddings.shape[0]), dtype=np.float32)
    cur_ranks = np.empty((embeddings.shape[0],), dtype=all_ranks.dtype)
    for cur_idx, cur_emb in enumerate(embeddings):
        dist = embeddings.dot(cur_emb)  # may be not normalized
        cur_ranks[dist.argsort()] = np.arange(len(dist))
        all_ranks[cur_idx, :] = cur_ranks / all_ranks.shape[0]

    def compare(left_path: Path, right_path: Path):
        return all_ranks[path_idx[left_path], path_idx[right_path]]

    return compare


def merge_ranks(predictions: Sequence[float], weights: Optional[np.ndarray] = None):
    if weights is None:
        return np.mean(predictions)
    else:
        weights = weights / np.sum(weights)
        return np.sum(np.array(predictions, copy=False) * weights)
