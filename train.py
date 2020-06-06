from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import mxnet as mx
import lib
from lib.data import load_info
from lib.basic_config import BasicConfig


def config_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('Training')
    parser.add_argument('data_path', type=str,
                        help='Path to the folder with images')
    return parser


def run_train(data_path: Path):
    cfg = BasicConfig(
        seed=444,
        name='arcface_ft',
        num_workers=6,
        gpus=(0,),
        batch_size=32,
        num_epochs=2,
        steps=(8, 14, 25, 35, 40, 50, 60, np.inf)
    )
    np.random.seed(cfg.seed)
    mx.random.seed(cfg.seed)
    train_df = load_info(data_path, Path('data/train_df.csv'))
    lib.train.train(cfg, train_df)


if __name__ == '__main__':
    args = config_parser().parse_args()
    run_train(Path(args.data_path))
