import numpy as np
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
    img_center = np.array([height / 2, width / 2], dtype=np.float32).reshape((1, 2))
    dist = np.linalg.norm(face_centers - img_center, axis=1)
    return int(np.argmin(dist))


def prepare_images(root_dir: Path,
                   output_dir: Path,
                   detector: Detector,
                   choose_face: Callable[[Scores, Bboxes, int, int], int]) -> None:
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
