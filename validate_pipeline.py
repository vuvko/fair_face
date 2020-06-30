from lib.compare import PipeMatcher, CompareModel
from lib.data import load_info, aggregate_subjects, sample_pairs
from lib.pipeline import mxnet_feature_extractor, pipeline_detector
from lib.matching import config_resnet_matcher, dummy_matcher
from lib.validate import plot_roc
from lib.submit import compare_all
import pandas as pd
from pathlib import Path
import numpy as np
import mxnet as mx
from more_itertools import unzip


def get_predicts(data_path, cache_dir, img_paths, pairs):
    img_matcher = config_resnet_matcher(img_paths, cache_dir)
    detector = pipeline_detector(img_paths, cache_dir / 'detector', small_face=16)
    experiment_names = ['ultimate5', 'test_center_vgg']
    epochs = [20, 10]
    experiment_path = Path('experiments')
    feature_extractors = [mxnet_feature_extractor(
        cache_dir / f'extractor_{cur_exp}_{cur_epoch:04d}',
        str(experiment_path / cur_exp / 'snapshots' / cur_exp),
        cur_epoch, use_flip=True, ctx=mx.gpu(0))
        for cur_exp, cur_epoch in zip(experiment_names, epochs)]
    comparator = PipeMatcher(img_paths, cache_dir, img_matcher, detector, feature_extractors)
    return np.array(list(unzip(compare_all(data_path, pairs, comparator))[2]))


def validate_pipe():
    cache_dir = Path('/run/media/andrey/Data/pipe_cache')
    data_path = Path('/run/media/andrey/Fast/FairFace/data/train/data')
    val_csv = Path('data') / 'val_df.csv'
    val_data = load_info(data_path, val_csv)
    num_sample = 1 * 10 ** 5
    subject_dict = aggregate_subjects(val_data['TEMPLATE_ID'], val_data['SUBJECT_ID'])
    sampled_pairs, sampled_labels = unzip(sample_pairs(subject_dict, num_sample))
    sampled_labels = np.array(list(sampled_labels))
    sampled_pairs = list(sampled_pairs)
    predictions = get_predicts(data_path, cache_dir, val_data['img_path'], sampled_pairs)
    plot_roc([(sampled_labels, predictions)], ['composite'],
             save_name='test_pipe.png')


if __name__ == '__main__':
    validate_pipe()
