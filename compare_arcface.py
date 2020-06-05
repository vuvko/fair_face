import numpy as np
import mxnet as mx
import argparse
from pathlib import Path
from lib.validate import validate, plot_roc
from lib.compare import CompareModel
from lib.metrics import cosine, euclidean


def config_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('Example of using validation')
    parser.add_argument('data_path', type=str,
                        help='Path to the folder with images')
    parser.add_argument('--num-sample', '-n', type=int, default=10 ** 3,
                        help='Number of samples to compare on')
    return parser


def compare(data_path: Path, num_sample: int) -> None:
    val_csv = Path('data') / 'val_df.csv'
    comparator = CompareModel(ctx=mx.gpu(0))
    comparator.metric = cosine
    cosine_res = validate(comparator, data_path, val_csv, num_sample=num_sample)
    comparator.metric = euclidean
    euclidean_res = validate(comparator, data_path, val_csv, num_sample=num_sample)
    plot_roc((cosine_res, euclidean_res), ('cosine', 'euclidean'), 'arcface_roc.png')


if __name__ == '__main__':
    args = config_parser().parse_args()
    compare(Path(args.data_path), args.num_sample)
