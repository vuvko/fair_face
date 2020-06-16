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
    parser.add_argument('experiment', type=str,
                        help='Experiment name')
    parser.add_argument('--num-sample', '-n', type=int, default=10 ** 3,
                        help='Number of samples to compare on')
    parser.add_argument('--use-flip', '-f', action='store_true', default=False,
                        help='Opt-in using flipped image for generating additional embedding.')
    return parser


def compare(data_path: Path, experiment: str, num_sample: int, use_flip: bool = False) -> None:
    val_csv = Path('data') / 'val_df.csv'
    model_path = Path('experiments') / experiment / 'snapshots'
    num_weights = len(list(model_path.iterdir())) - 1
    results = []
    for cur_epoch in range(num_weights):
        comparator = CompareModel(str(model_path / experiment), cur_epoch + 1, use_flip=use_flip, ctx=mx.gpu(0))
        comparator.metric = cosine
        cosine_res = validate(comparator, data_path, val_csv, num_sample=num_sample)
        results.append(cosine_res)
    plot_roc(results, save_name=f'{experiment}_roc.png')


if __name__ == '__main__':
    args = config_parser().parse_args()
    compare(Path(args.data_path), args.experiment, args.num_sample, args.use_flip)
