import numpy as np
import mxnet as mx
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from insightface import model_zoo
from sklearn import preprocessing
import os
from queue import Queue
from . import metrics
from .dataset import ImgDataset
from .utils import calc_closest_k, label_embeddings

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


class PipeMatcher:

    def __init__(self, all_imgs, cache_dir: Path, img_matcher, detector, feature_extractors, k_closest: int = 3):
        self.cache_dir = cache_dir
        if not cache_dir.exists():
            os.makedirs(str(cache_dir))
        self.all_imgs = all_imgs
        self.img_matcher = img_matcher
        self.detector = detector
        self.feature_extractors = feature_extractors
        self.k_closest = k_closest
        self.path2idx = {}
        self.close_match = None
        self.calc_closest_match()
        self.features = {}
        self.prepare_features()
        self.final_ranks = None
        self.path2idx = {cur_path: cur_idx for cur_idx, cur_path in enumerate(self.all_imgs)}
        self.ensemble_ranks()

    def calc_closest_match(self):
        viewed_pairs = np.zeros((len(self.all_imgs), len(self.all_imgs)), dtype=np.bool)
        connected = np.zeros((len(self.all_imgs), len(self.all_imgs)), dtype=np.bool)
        self.path2idx = {p: i for i, p in enumerate(self.all_imgs)}
        print('Finding close images')
        to_view = list(enumerate(self.all_imgs))
        prog_bar = tqdm(total=len(to_view))
        while len(to_view) > 0:
            outer_idx, outer_path = to_view[0]
            for inside_idx, inside_path in to_view[1:]:
                if viewed_pairs[outer_idx, inside_idx]:
                    continue
                viewed_pairs[outer_idx, inside_idx] = 1
                viewed_pairs[inside_idx, outer_idx] = 1
                if self.img_matcher(outer_path, inside_path):
                    connected[outer_idx, inside_idx] = 1
                    connected[inside_idx, outer_idx] = 1
                    break
            to_view = to_view[1:]
            prog_bar.update(1)
        self.close_match = connected

        # if (self.cache_dir / 'labeling.csv').exists():
        #     df = pd.read_csv(self.cache_dir / 'labeling.csv')
        #     for img_path, close_match in zip(df['img_path'], df['close_match']):
        #         self.close_match[img_path] = close_match
        # else:
        #     with open(str(self.cache_dir / 'labeling.csv'), 'w') as f:
        #         print('img_path,close_match', file=f)
        # print('Finding close images')
        # for cur_idx, anchor_path in enumerate(tqdm(self.all_imgs[:-1])):
        #     if self.close_match.get(anchor_path, None) is not None:
        #         continue
        #     self.close_match[anchor_path] = None
        #     for inside_idx, comp_path in zip(range(cur_idx + 1, len(self.all_imgs)), self.all_imgs[cur_idx + 1:]):
        #         if self.img_matcher(anchor_path, comp_path):
        #             self.close_match[anchor_path] = comp_path
        #             self.close_match[comp_path] = anchor_path
        #             with open(str(self.cache_dir / 'labeling.csv'), 'a') as f:
        #                 print(f'{anchor_path},{comp_path}', file=f)
        #                 print(f'{comp_path},{anchor_path}', file=f)
        #             break

    def collect_paths(self, cur_path: Path):
        # collected_paths = {cur_path}
        # view_queue = Queue()
        # view_queue.put(cur_path)
        # while not view_queue.empty():
        #     cur_path = view_queue.get()
        #     collected_paths.update(set(close_match[cur_path]))
        #     cur_path = close_match[cur_path]
        # return collected_paths
        # collected_idx = np.nonzero(self.close_match[self.path2idx[cur_path]])[0]
        view_queue = Queue()
        first_idx = self.path2idx[cur_path]
        collected_idx = set()
        view_queue.put(first_idx)
        while not view_queue.empty():
            cur_idx = view_queue.get()
            if cur_idx in collected_idx:
                continue
            collected_idx.add(cur_idx)
            next_idx = np.nonzero(self.close_match[cur_idx])[0]
            if len(next_idx) > 0:
                [view_queue.put(i) for i in next_idx]
        return [self.all_imgs[i] for i in collected_idx]

    def aggregate_features(self):
        to_aggregate = set(self.all_imgs)
        while len(to_aggregate) > 0:
            cur_path = to_aggregate.pop()
            collected_paths = self.collect_paths(cur_path)
            to_aggregate = to_aggregate - set(collected_paths)
            cur_features = [self.features[k] for k in collected_paths]
            aggregated = []
            for cur_idx in range(len(self.feature_extractors)):
                cur_aggregated = np.mean([cur_embs[cur_idx] for cur_embs in cur_features], axis=0)
                aggregated.append(cur_aggregated / np.sqrt(np.sum(cur_aggregated ** 2, axis=-1, keepdims=True)))
            for cur_ins in collected_paths:
                self.features[cur_ins] = aggregated

    def prepare_features(self):
        for cur_path in self.all_imgs:
            prep_img = self.detector(cur_path)
            cur_features = []
            for cur_idx, cur_extractor in enumerate(self.feature_extractors):
                cur_features.append(cur_extractor(prep_img, cur_path))
            self.features[cur_path] = np.array(cur_features)
        self.aggregate_features()

    def ensemble_ranks(self):
        all_features = np.array([self.features[k] for k in self.all_imgs]).transpose((1, 0, 2))
        num_imgs = len(self.all_imgs)
        all_ranks = np.empty((all_features.shape[0], num_imgs, num_imgs), dtype=np.float32)
        cur_ranks = np.empty((num_imgs,), dtype=all_ranks.dtype)
        for cur_idx, cur_features in enumerate(all_features):
            for inside_idx, cur_emb in enumerate(cur_features):
                dist = cur_features.dot(cur_emb)  # may be not normalized
                cur_ranks[dist.argsort()] = np.arange(num_imgs)
                all_ranks[cur_idx, inside_idx, :] = cur_ranks / num_imgs
        self.final_ranks = np.mean(all_ranks, axis=0)

    def __call__(self, left_path: Path, right_path: Path) -> float:
        return self.final_ranks[self.path2idx[left_path], self.path2idx[right_path]]


