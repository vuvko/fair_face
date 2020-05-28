import numpy as np
from pathlib import Path
from skimage import io
import pandas as pd
from .mytypes import DataInfo, Img
from typing import Optional


def load_info(data_dir: Path, csv_path: Path) -> pd.DataFrame:
    info = pd.read_csv(str(csv_path), delimiter=',')
    img_paths = []
    for template_id in info['template_id']:
        img_paths.append(data_dir / f'{template_id}.jpg')
    info['img_path'] = img_paths
    return info


def load_img(img_path: Path) -> Optional[Img]:
    return io.imread(str(img_path))
