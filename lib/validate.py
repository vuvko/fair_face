import numpy as np
import pandas as pd
import mxnet as mx
from mxnet.gluon.data import DataLoader
from pathlib import Path
from more_itertools import unzip
from itertools import count
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from .data import aggregate_subjects, sample_pairs
from .submit import compare_all
from .mytypes import Comparator
from .basic_config import BasicConfig
from .dataset import InfoDataset
from typing import Iterable, Tuple, Optional


def validate(comparator: Comparator,
             data_dir: Path,
             validation_csv: Path,
             num_sample: int = 10 ** 3
             ) -> Tuple[np.ndarray, np.ndarray]:
    df = pd.read_csv(validation_csv)
    subject_dict = aggregate_subjects(df['TEMPLATE_ID'], df['SUBJECT_ID'])
    sampled_pairs, sampled_labels = unzip(sample_pairs(subject_dict, num_sample))
    sampled_labels = np.array(list(sampled_labels))
    predictions = np.array(list(unzip(compare_all(data_dir, sampled_pairs, comparator))[2]))
    return sampled_labels, predictions


def plot_roc(results: Iterable[Tuple[np.ndarray, np.ndarray]],
             experiment_names: Optional[Iterable[str]] = None,
             save_name: str = 'cur_results.png'
             ) -> None:
    if experiment_names is None:
        experiment_names = count()
    for cur_name, (cur_labels, cur_preds) in zip(experiment_names, results):
        fpr, tpr, _ = roc_curve(cur_labels, cur_preds)
        plt.plot(fpr, 1 - tpr, label=f'{cur_name} AUC: {roc_auc_score(cur_labels, cur_preds):.5f}')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('False Positive Rate')
    plt.ylabel('False Negative Rate')
    plt.grid()
    plt.legend()
    plt.savefig(save_name)
    plt.show()


def run_model(model_prefix: str, model_epoch: int, config: BasicConfig, data_df: pd.DataFrame, save_path: Path):
    use_gpu = len(config.gpus) > 0
    if use_gpu:
        ctx = [mx.gpu(cur_idx) for cur_idx in config.gpus]
    else:
        ctx = [mx.cpu()]
    sym, args, auxs = mx.model.load_checkpoint(model_prefix, model_epoch)
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    data_shape = (1, 3, 112, 112)
    model.bind(data_shapes=[('data', data_shape)], for_training=False)
    model.set_params(args, auxs)
    dataset = InfoDataset(data_df, filter_fn=config.filter_fn, augs=config.test_augmentations)
    data = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        sampler=None,
        num_workers=config.num_workers,
        pin_memory=use_gpu
    )
    predictions = []
    all_paths, labels = unzip(dataset.data)
    for i, batch in tqdm(enumerate(data), total=len(data)):
        data = mx.gluon.utils.split_and_load(batch[0], ctx_list=ctx, even_split=False)
        batch = mx.io.DataBatch(data)
        model.forward(batch, is_train=False)
        predictions.append(model.get_outputs()[0].asnumpy())
    predictions = np.concatenate(predictions, axis=0)
    labels = np.array(list(labels))
    all_paths = list(all_paths)
    np.savez(str(save_path), paths=all_paths, labels=labels, preds=predictions)
    return all_paths, labels, predictions
