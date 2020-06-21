from lib.validate import validate, plot_roc
from lib.compare import CompareModel
from lib.data import load_info, sample_pairs, aggregate_subjects
from lib import metrics
import mxnet as mx
import numpy as np
from tqdm import tqdm
from pathlib import Path
from more_itertools import unzip
import argparse
from typing import Sequence


def report(data_path: Path, experiments: Sequence[str]):
    val_csv = Path('data') / 'val_df.csv'
    df = load_info(data_path, val_csv)
    num_sample = 10 ** 5
    subject_dict = aggregate_subjects(df['TEMPLATE_ID'], df['SUBJECT_ID'])
    sampled_pairs, sampled_labels = unzip(sample_pairs(subject_dict, num_sample))
    sampled_labels = np.array(list(sampled_labels))
    sampled_pairs = list(sampled_pairs)
    results = []
    for cur_exp in experiments:
        model_path = Path('experiments') / cur_exp / 'snapshots'
        cur_epoch = len(list(model_path.iterdir())) - 1
        comparator = CompareModel(str(model_path / cur_exp), cur_epoch, use_flip=True, ctx=mx.gpu(0))
        cur_preds = validate(comparator, data_path, val_csv, num_sample=0, pairs=sampled_pairs, labels=sampled_labels)[1]
        results.append(cur_preds)
    results = np.array(results)
    plot_roc(((sampled_labels, cur_res) for cur_res in results), experiments, save_name=f'report_roc.png')
    # calculating -1's part in our AUC
    f_result = results[0]
    positive_part = np.sum(np.abs(f_result[sampled_labels > 0.5] - -1) < 1e-7)
    negative_part = np.sum(np.abs(f_result[sampled_labels < 0.5] - -1) < 1e-7)
    print(f'Positive part: {positive_part}')
    print(f'Negative part: {negative_part}')
    # generating common errors
    pos_thresh = 0.3
    neg_thresh = 0.0
    positives = results > pos_thresh
    negatives = results < neg_thresh
    common_positive = [idx for idx, cur_row in enumerate(positives.T) if np.sum(cur_row) > 0.4 * len(cur_row)]
    common_negative = [idx for idx, cur_row in enumerate(negatives.T) if np.sum(cur_row) > 0.4 * len(cur_row)]
    common_pos_errors = np.where(sampled_labels[np.array(common_positive, dtype=np.int32)] < 0.5)[0]
    common_neg_errors = np.where(sampled_labels[np.array(common_negative, dtype=np.int32)] > 0.5)[0]
    with open('common_pos_err.txt', 'w') as f:
        for cur_idx in common_pos_errors:
            print(f'"{sampled_pairs[cur_idx][0]}", "{sampled_pairs[cur_idx][1]}"', file=f)
    with open('common_neg_err.txt', 'w') as f:
        for cur_idx in common_neg_errors:
            print(f'"{sampled_pairs[cur_idx][0]}", "{sampled_pairs[cur_idx][1]}"', file=f)


def config_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('Preparing a report')
    parser.add_argument('data_path', type=str,
                        help='Path to the folder with images')
    parser.add_argument('experiments', metavar='N', type=str, nargs='+')
    return parser


if __name__ == '__main__':
    args = config_parser().parse_args()
    report(Path(args.data_path), args.experiments)
