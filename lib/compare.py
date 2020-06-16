import numpy as np
import mxnet as mx
from pathlib import Path
from insightface import model_zoo
from . import metrics
from .dataset import ImgDataset

from .mytypes import Embedding, Labels
from typing import List


class CompareModel(object):

    def __init__(self,
                 model_name: str = 'arcface_r100_v1',
                 epoch_num: int = 0,
                 use_flip: bool = False,
                 ctx: mx.Context = mx.cpu()):
        self.use_flip = use_flip
        self.embeddings = dict()
        if model_name == 'arcface_r100_v1':
            model = model_zoo.get_model(model_name)
            if ctx.device_type.startswith('cpu'):
                ctx_id = -1
            else:
                ctx_id = ctx.device_id
            model.prepare(ctx_id=ctx_id)
            self.model = model.model
        else:
            sym, arg_params, aux_params = mx.model.load_checkpoint(model_name, epoch_num)
            sym = sym.get_internals()['fc1_output']
            model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
            data_shape = (2 if use_flip else 1, 3, 112, 112)
            model.bind(data_shapes=[('data', data_shape)], for_training=False)
            model.set_params(arg_params, aux_params)
            #warmup
            data = mx.nd.zeros(shape=data_shape)
            db = mx.io.DataBatch(data=(data,))
            model.forward(db, is_train=False)
            embedding = model.get_outputs()[0].asnumpy()
            self.model = model
        self.metric = metrics.cosine

    def get_embedding(self, im_path: Path) -> Embedding:
        if im_path not in self.embeddings:
            img = mx.img.imread(str(im_path))
            prep_img = img.transpose((2, 0, 1)).expand_dims(0).astype(np.float32)
            if self.use_flip:
                flipped = mx.nd.flip(img, axis=1).transpose((2, 0, 1)).expand_dims(0).astype(np.float32)
                img = mx.nd.concatenate([prep_img, flipped])
            else:
                img = prep_img
            batch = mx.io.DataBatch([img])
            self.model.forward(batch, is_train=False)
            self.embeddings[im_path] = self.model.get_outputs()[0].asnumpy().mean(axis=0)
        return self.embeddings[im_path]

    def prepare_all_embeddings(self,
                               paths: List[Path],
                               **loader_params) -> None:
        data = mx.gluon.data.DataLoader(ImgDataset(paths), shuffle=False, **loader_params)
        raise NotImplementedError

    def __call__(self, path1: Path, path2: Path) -> float:
        emb1 = self.get_embedding(path1)
        emb2 = self.get_embedding(path2)
        return self.metric(emb1, emb2)
