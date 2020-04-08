import numpy as np
from pathlib import Path
import logging
import os
import subprocess
from tqdm import tqdm
from .utils import csv_generator
from .compare import CompareModel

from .mytypes import Comparator
from typing import Iterable, Tuple, Generator

DEFINITELY_NOT_SAME = -1


def compare_all(data_dir: Path,
                pairs: Iterable[Tuple[int, int]],
                comparator: Comparator
                ) -> Generator[Tuple[int, int, float], None, None]:
    for left_el, right_el in tqdm(pairs):
        left_path = data_dir / f'{left_el}.jpg'
        right_path = data_dir / f'{right_el}.jpg'
        if not left_path.exists() or not right_path.exists():
            yield left_el, right_el, DEFINITELY_NOT_SAME
            continue
        yield left_el, right_el, comparator(left_path, right_path)


def submit(val_dir: Path,
           val_csv: Path,
           comparator: Comparator,
           submit_name: str = 'test',
           submit_dir: Path = Path('submits')
           ) -> None:
    pairs = csv_generator(val_csv)
    cur_submit_dir = submit_dir / submit_name
    if cur_submit_dir.exists():
        logging.error('Submission directory already exists, aborting.')
        return None
    os.makedirs(str(cur_submit_dir))
    with open(str(cur_submit_dir / 'predictions.csv'), 'w') as f:
        print('TEMPLATE_ID1,TEMPLATE_ID2,SCORE', file=f)
        for left_el, right_el, score in compare_all(val_dir, pairs, comparator):
            print(f'{left_el},{right_el},{score}', file=f)
