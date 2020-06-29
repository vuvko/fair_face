import numpy as np
import mxnet as mx
import argparse
from pathlib import Path
from more_itertools import unzip
from lib.validate import validate, plot_roc
from lib.data import load_info, sample_pairs, aggregate_subjects
from lib.compare import CompareModel, config_rank_comparator, merge_ranks
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
    val_csv = Path('data') / 'wide_val.csv'
    model_path = Path('experiments') / experiment / 'snapshots'
    num_weights = len(list(model_path.iterdir())) - 1
    results = []
    df = load_info(data_path, val_csv)
    exists = [idx for idx, cur_path in enumerate(df['img_path']) if cur_path.exists()]
    val_data = df.iloc[np.array(exists)]
    subject_dict = aggregate_subjects(df['TEMPLATE_ID'], df['SUBJECT_ID'])
    sampled_pairs, sampled_labels = unzip(sample_pairs(subject_dict, num_sample))
    sampled_labels = np.array(list(sampled_labels))
    sampled_pairs = list(sampled_pairs)
    names = []
    rank_results = []
    for cur_epoch in range(7, num_weights):
        comparator = CompareModel(str(model_path / experiment), cur_epoch + 1, use_flip=use_flip, ctx=mx.gpu(0))
        comparator.metric = cosine
        rank_comparator = config_rank_comparator(comparator, val_data['img_path'])
        cosine_res = validate(comparator, data_path, val_csv, num_sample=0, pairs=sampled_pairs, labels=sampled_labels)
        rank_results.append(validate(rank_comparator, data_path, val_csv, num_sample=0,
                                     pairs=sampled_pairs, labels=sampled_labels)[1])
        results.append(cosine_res)
        names.append(f'epoch {cur_epoch + 1:04d}')
        results.append((sampled_labels, rank_results[-1]))
        names.append(f'epoh {cur_epoch + 1:04d} rank')
    rank_merge_results = np.mean(rank_results, axis=0)
    results.append((sampled_labels, rank_merge_results))
    names.append('merged')

    plot_roc(results, experiment_names=names, save_name=f'{experiment}_{"flip" if use_flip else "no_flip"}_roc.png')


if __name__ == '__main__':
    args = config_parser().parse_args()
    compare(Path(args.data_path), args.experiment, args.num_sample, args.use_flip)
