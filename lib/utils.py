import numpy as np
import mxnet as mx
from insightface.utils.face_align import norm_crop
from pathlib import Path
import shutil
import os
from tqdm import tqdm
import cv2
import imghdr
import logging
from queue import Queue
import pickle
from .basic_config import BasicConfig

from .mytypes import Bboxes, Scores, Landmarks, Detector
from typing import Generator, Callable, Tuple, Optional


def ensure_path(cur: Path) -> Path:
    if not cur.exists():
        os.makedirs(str(cur))
    return cur


def is_valid_img(img_path: Path) -> bool:
    valid_response = ['jpeg', 'png', 'bmp']
    return imghdr.what(str(img_path)) in valid_response


def img_generator(dataset_root: Path) -> Generator[Path, None, None]:
    dir_queue = Queue()
    dir_queue.put(dataset_root)
    while not dir_queue.empty():
        cur_root = dir_queue.get()
        for file_path in cur_root.iterdir():
            if file_path.is_dir():
                dir_queue.put(file_path)
                continue
            if not is_valid_img(file_path):
                continue
            yield file_path


def csv_generator(csv_path: Path) -> Generator[Tuple[int, int], None, None]:
    with open(str(csv_path), 'r') as f:
        f.readline()
        for line in f:
            left_el, right_el, score = line.strip().split(',')
            yield int(left_el), int(right_el)


def choose_center_face(scores: Scores, bboxes: Bboxes, width: int, height: int) -> int:
    face_centers = 0.5 * (bboxes[:, :2] + bboxes[:, 2:])
    img_center = np.array([width / 2, height / 2], dtype=np.float32).reshape((1, 2))
    dist = np.linalg.norm(face_centers - img_center, axis=1)
    return int(np.argmin(dist))


def prepare_images(root_dir: Path,
                   output_dir: Path,
                   detector: Detector,
                   choose_face: Callable[[Scores, Bboxes, int, int], int],
                   small_face: int = -1
                   ) -> None:
    logging.info(f'Starting preparation: input dir {root_dir} output dir {output_dir}')
    img_paths = list(img_generator(root_dir))
    for img_path, (scores, bboxes, landmarks) in zip(img_paths, tqdm(detector(img_paths), total=len(img_paths))):
        new_path = output_dir / img_path.relative_to(root_dir)
        ensure_path(new_path.parent)
        img = cv2.imread(str(img_path))
        height, width = img.shape[:2]
        if len(landmarks) < 1:
            logging.warning(f'smth wrong with {img_path}')
            continue
        face_idx = choose_face(scores, bboxes, width, height)
        if small_face > 0:
            bbox = bboxes[face_idx]
            if bbox[2] - bbox[0] < small_face or bbox[3] - bbox[1] < small_face:
                logging.warning(f'small face in image {img_path}')
                continue
        warped_img = norm_crop(img, landmarks[face_idx])
        cv2.imwrite(str(new_path), warped_img)


def prepare_experiment(config: BasicConfig) -> Optional[Tuple[Path, logging.Logger]]:
    experiment_path = Path('experiments') / config.name
    if experiment_path.exists():
        logging.error('Experiment was already created, aborting.')
        return None
    os.makedirs(str(experiment_path))
    with open(str(experiment_path / 'config.pkl'), 'wb') as f:
        pickle.dump(config, f)
    logger = logging.getLogger(f'{config.name} experiment')
    fh = logging.FileHandler(str(experiment_path / 'experiment.log'))
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(logging.Formatter('%(asctime)s [%(name)s]:%(levelname)s: %(message)s'))
    logger.addHandler(fh)
    logger.setLevel(logging.DEBUG)
    return experiment_path, logger


def calc_closest_k(embs, k: int = 2, batch_size: int = 10 ** 2, use_gpu: bool = True):
    closest_vals = np.empty((embs.shape[0], k), dtype=np.float32)
    closest_idx = np.empty((embs.shape[0], k), dtype=np.int32)
    if use_gpu:
        ctx = mx.gpu(0)
    else:
        ctx = mx.cpu()
    mx_embeddings = mx.nd.array(embs.astype(np.float32), ctx=ctx)
    mx_embeddings = mx_embeddings / mx.nd.sqrt(mx.nd.sum(mx_embeddings ** 2, axis=-1, keepdims=True))
    for bg in tqdm(range(0, embs.shape[0], batch_size)):
        ed = min(embs.shape[0], bg + batch_size)
        cur_embs = mx_embeddings[bg:ed]
        cur_dist = 1 - mx.nd.dot(cur_embs, mx_embeddings.T)
        cur_vals, cur_idx = mx.nd.topk(cur_dist, k=k, axis=-1, ret_typ='both', dtype='int32', is_ascend=True)
        closest_vals[bg:ed] = cur_vals.asnumpy()
        closest_idx[bg:ed] = cur_idx.asnumpy()
    return closest_vals, closest_idx


def label_embeddings(closest_vals, closest_idx, threshold: float = 0.3):
    NO_LABEL = -1
    labels = np.zeros((closest_vals.shape[0],), dtype=np.int32) + NO_LABEL
    cur_label = 0
    # closest_vals, closest_idx = calc_closest_k(embs, k=k, use_gpu=use_gpu)
    for cur_idx, (cur_dists, cur_closest) in tqdm(enumerate(zip(closest_vals, closest_idx))):
        if labels[cur_idx] != NO_LABEL:
            continue
        to_label = cur_closest[cur_dists < threshold]
        if np.all(labels[to_label]) != NO_LABEL:
            search_labels = np.unique(labels[to_label])
            more_to_label = [to_label]
            for cur_l in search_labels:
                if cur_l == NO_LABEL:
                    continue
                more_to_label.append(np.where(labels == cur_l)[0])
            to_label = np.concatenate(more_to_label, axis=0)
        labels[to_label] = cur_label
        cur_label += 1
    return labels
