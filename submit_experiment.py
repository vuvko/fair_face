import numpy as np
import mxnet as mx
import argparse
from pathlib import Path
from lib.validate import validate, plot_roc
from lib.compare import CompareModel
from lib.submit import submit
from lib.metrics import cosine, euclidean


def config_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('Example of using validation')
    parser.add_argument('data_path', type=str,
                        help='Path to the folder with images.')
    parser.add_argument('csv_path', type=str,
                        help='Path to the testing csv file.')
    parser.add_argument('experiment', type=str,
                        help='Experiment name')
    parser.add_argument('epoch', type=int,
                        help='Epoch number')
    return parser


def compare(data_path: Path, csv_path: Path, experiment: str, epoch: int) -> None:
    model_path = Path('experiments') / experiment / 'snapshots'
    comparator = CompareModel(str(model_path / experiment), epoch, ctx=mx.gpu(0))
    comparator.metric = cosine
    submit(data_path, csv_path, comparator, f'{experiment}_{epoch}')


if __name__ == '__main__':
    args = config_parser().parse_args()
    compare(Path(args.data_path), Path(args.csv_path), args.experiment, args.epoch)
