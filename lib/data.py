import numpy as np
from pathlib import Path
from skimage import io
import pandas as pd
from pathos.multiprocessing import Pool
import random
from .mytypes import DataInfo, Img, SubjectDict, IdPair, Label
from typing import Optional, Sequence, Tuple, Iterable


def load_info(data_dir: Path, csv_path: Path) -> pd.DataFrame:
    info = pd.read_csv(str(csv_path), delimiter=',')
    img_paths = []
    for template_id in info['TEMPLATE_ID']:
        img_paths.append(data_dir / f'{template_id}.jpg')
    info['img_path'] = img_paths
    return info


def load_img(img_path: Path) -> Optional[Img]:
    return io.imread(str(img_path))


def sample_triplet(subjects: SubjectDict, subject_ids, cur_subject) -> Tuple[int, int, int]:
    # sample an anchor with a positive candidate
    positive_pair = np.random.choice(subjects[cur_subject], size=2, replace=False)
    # sample a negative_candidate
    other_subjects = subject_ids - {cur_subject}
    negative_subject = random.choice(list(other_subjects))
    negative_candidate = random.choice(subjects[negative_subject])
    return positive_pair[0], positive_pair[1], negative_candidate


def sample_triplets(subjects: SubjectDict, num_sample: int = 10 ** 3) -> Iterable[Tuple[int, int, int]]:
    subject_ids = set(subjects.keys())
    sample_subjects = np.random.choice(list(subject_ids), size=num_sample)
    # triplets = []
    # for cur_subject in sample_subjects:
    #     triplets.append(sample_triplet(subjects, subject_ids, cur_subject))
    with Pool(4) as p:
        triplets = p.map(lambda cur_subject: sample_triplet(subjects, subject_ids, cur_subject), sample_subjects)
    return triplets


def sample_pairs(subjects: SubjectDict, num_sample: int = 2 * 10 ** 3) -> Sequence[Tuple[IdPair, Label]]:
    pairs = []
    for anchor, positive, negative in sample_triplets(subjects, num_sample=num_sample // 2):
        pairs.append(((anchor, positive), 1))
        pairs.append(((anchor, negative), 0))
    return pairs


def aggregate_subjects(template_ids: Sequence[int], subject_ids: Sequence[int]) -> SubjectDict:
    subjects = {}
    for cur_template, cur_subject in zip(template_ids, subject_ids):
        if cur_subject not in subjects:
            subjects[cur_subject] = []
        subjects[cur_subject].append(cur_template)
    return subjects


def split_data(subjects: SubjectDict, split_ratio: float) -> Tuple[SubjectDict, SubjectDict]:
    all_subjects = list(subjects.keys())
    random.shuffle(all_subjects)
    left_subjects = all_subjects[:int(split_ratio * len(all_subjects))]
    right_subjects = all_subjects[int(split_ratio * len(all_subjects)):]
    return {key: subjects[key] for key in left_subjects}, {key: subjects[key] for key in right_subjects}
