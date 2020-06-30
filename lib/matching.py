from pathlib import Path
import numpy as np
import mxnet as mx
from mxnet import gluon
import gluoncv
from mxnet.gluon.data.vision import transforms
from tqdm import tqdm
import subprocess
import os
from typing import Sequence, Tuple


def collect_matches(superglue_path: Path, cache_dir: Path,
                    input_dir: Path, input_pairs: Sequence[Tuple[int, int]],
                    conf_thresh: float = 0.512, min_matched: int = 52):
    with open(str(cache_dir / 'pairs.txt'), 'w') as f:
        for left_item, right_item in input_pairs:
            print(f'{left_item}.jpg {right_item}.jpg', file=f)
    input_dir_relative = input_dir.relative_to(superglue_path)
    command = f'cd {superglue_path} && python match_pairs.py --input_dir {input_dir_relative} ' \
              f'--input_pairs {cache_dir.relative_to(superglue_path) / "pairs.txt"} ' \
              f'--output_dir {cache_dir.relative_to(superglue_path)} --cache'
    os.system(command)
    results = []
    for left_item, right_item in input_pairs:
        path = cache_dir / f'{left_item}_{right_item}_matches.npz'
        f = np.load(str(path))
        conf = f['match_confidence']
        results.append(np.sum(conf > conf_thresh) > min_matched)
    return results


class Loader(gluon.data.Dataset):

    def __init__(self, lst):
        super().__init__()
        self.lst = lst

    def __len__(self):
        return len(self.lst)

    def __getitem__(self, idx):
        path = self.lst[idx]
        return mx.img.imread(str(path))


def config_resnet_matcher(all_imgs, conf_thresh: float = 0.92557, ctx=mx.gpu(0)):
    mod = gluoncv.model_zoo.get_model('ResNet50_v2', ctx=ctx, pretrained=True)
    mod.fc = gluon.contrib.nn.Identity()
    mod.hybridize()
    transform_fn = transforms.Compose([
        transforms.Resize(256, keep_ratio=True),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    embs = []
    path2idx = {cp: i for i, cp in enumerate(all_imgs)}
    dataset = Loader(all_imgs)
    loaded = gluon.data.DataLoader(
        dataset.transform_first(transform_fn),
        batch_size=64,
        shuffle=False,
        num_workers=3,
        pin_memory=True
    )
    print('Preparing matcher')
    for batch in tqdm(loaded, total=len(loaded)):
        inp = batch.as_in_context(mx.gpu(0))
        pred = mod(inp)
        embs.append(pred.asnumpy())
    embs = np.concatenate(embs, axis=0)
    normed = embs / np.sqrt(np.sum(embs ** 2, axis=-1, keepdims=True))

    def match(left: Path, right: Path):
        dist = np.dot(normed[path2idx[left]], normed[path2idx[right]])
        return dist > conf_thresh

    return match
