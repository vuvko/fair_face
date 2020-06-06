import numpy as np
import pandas as pd
from pathlib import Path
from more_itertools import unzip
from itertools import count
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score
from .data import aggregate_subjects, sample_pairs
from .submit import compare_all
from .mytypes import Comparator
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
    plt.figure()
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
