from dataclasses import dataclass
from collections import defaultdict
import albumentations as alb
import numpy as np
import cv2
from pathlib import Path
from .filter import ComposeFilter, ExistsFilter
from .mytypes import KwArgs, ItemFilter
from typing import List, Tuple, Optional, Sequence, Callable, Any


@dataclass
class BasicConfig:

    seed: int = 100
    name: str = 'Untitled'

    train_augmentations: alb.Compose = alb.Compose([
        alb.Resize(112, 112)
    ])
    test_augmentations: alb.Compose = alb.Compose([
        alb.Resize(112, 112)
    ])

    initial_lr: float = 1e-4
    momentum: float = 0.9
    weight_decay: float = 1e-4
    steps: Sequence[int] = (np.inf,)

    num_workers: int = 0
    gpus: Sequence[int] = ()
    batch_size: int = 1
    num_epochs: int = 10
    save_epoch: int = 1

    lr_factor: float = 0.75
    warmup_epoch: int = 1
    cooldown_epoch: int = 1
    clip_gradient: float = 1.0

    pretrained_experiment: Optional[str] = None
    pretrain_args: KwArgs = defaultdict

    filter_fn: ItemFilter = ExistsFilter()
