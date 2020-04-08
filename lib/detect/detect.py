import numpy as np
import mxnet as mx
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data.vision import transforms
from pathlib import Path

from .dataset import PrepareDataset
from .transforms import ResizeLong, NonNormalizedTensor
from ..mytypes import Detector, Detections
from typing import Sequence, Generator


def get_retina_det(prefix: Path,
                   epoch: int = 0,
                   resolution: int = 256,
                   batch_size: int = 1,
                   num_workers: int = 0,
                   ctx: mx.Context = mx.cpu()
                   ) -> Detector:
    sym, arg_params, aux_params = mx.model.load_checkpoint(str(prefix), epoch)
    model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
    data_shape = (1, 3, resolution, resolution)
    model.bind(data_shapes=[('data', data_shape)], for_training=False)
    model.set_params(arg_params, aux_params)
    # warmup
    data = mx.nd.zeros(data_shape)
    model.forward(mx.io.DataBatch(data=(data,)), is_train=False)
    model.get_outputs()[0].asnumpy()
    # end of warmup
    detector_transform = transforms.Compose([
        ResizeLong(resolution),
        NonNormalizedTensor()
    ])

    def prepare_data(img_list: Sequence[Path]) -> DataLoader:
        data = DataLoader(PrepareDataset(img_list).transform_first(detector_transform),
                          num_workers=num_workers,
                          batch_size=batch_size,
                          pin_memory=ctx.device_type == 'gpu',
                          last_batch='keep')
        return data

    def detect(img_list: Sequence[Path]) -> Generator[Detections, None, None]:
        data = prepare_data(img_list)
        for img_batch, idx_batch, width_batch, height_batch in data:
            model.forward(mx.io.DataBatch(data=(img_batch.as_in_context(ctx),)), is_train=False)
            results = model.get_outputs()[0].asnumpy()
            scores = results[:, :, 1]
            bboxes = results[:, :, 2:6]
            landmarks = results[:, :, 6:]
            num_landmarks = landmarks.shape[2] // 2
            landmarks = landmarks.reshape((scores.shape[0], scores.shape[1], num_landmarks, 2))
            for img_idx, width, height, cur_scores, cur_bboxes, cur_landmarks in \
                    zip(idx_batch, width_batch, height_batch, scores, bboxes, landmarks):
                max_size = max(width.asnumpy(), height.asnumpy())
                filtered_idx = cur_scores > 1e-6
                cur_scores = cur_scores[filtered_idx]
                cur_bboxes = cur_bboxes[filtered_idx] * max_size
                cur_landmarks = cur_landmarks[filtered_idx] * max_size
                yield cur_scores, cur_bboxes, cur_landmarks

    return detect


def get_retina_mnet(*args, resolution: int = 256, **kwargs):
    return get_retina_det(Path.home() / 'models' / f'det_mnet025_{resolution}', *args,
                          resolution=resolution, **kwargs)


def get_retina_resnet50(*args, resolution: int = 512, **kwargs):
    return get_retina_det(Path.home() / 'models' / f'det_r50_{resolution}', *args, resolution=resolution, **kwargs)
