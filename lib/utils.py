import numpy as np
import mxnet as mx
from mxnet.gluon.data.vision import transforms
from insightface.utils.face_align import norm_crop
from insightface import model_zoo
from pathlib import Path
import shutil
import os
from tqdm import tqdm
import cv2
import imghdr
import logging
from queue import Queue

from .mytypes import Bboxes, Scores, Landmarks, Detector
from typing import Generator, Callable, Tuple


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
        if not new_path.parent.exists():
            os.makedirs(str(new_path.parent))
        img = cv2.imread(str(img_path))
        height, width = img.shape[:2]
        if len(landmarks) < 1:
            logging.warning(f'smth wrong with {img_path}')
            continue
        face_idx = choose_face(scores, bboxes, width, height)
        warped_img = norm_crop(img, landmarks[face_idx])
        cv2.imwrite(str(new_path), warped_img)
