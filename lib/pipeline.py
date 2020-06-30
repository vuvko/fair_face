from pathlib import Path
import numpy as np
import mxnet as mx
from mxnet import gluon
import xxhash
from mxnet.gluon.data.vision import transforms
import cv2
from tqdm import tqdm
from insightface import model_zoo
from insightface.utils.face_align import norm_crop
from .utils import choose_center_face
from .detect import get_retina_resnet50
from .mytypes import Detections, Detector
from typing import Sequence, Generator


def get_detector_with_backup() -> Detector:
    detector = get_retina_resnet50(resolution=512, ctx=mx.gpu(0), batch_size=8, num_workers=6)
    backup = model_zoo.get_model('retinaface_r50_v1')
    backup.prepare(ctx_id=0, nms=0.4)

    def detect(img_paths: Sequence[Path]) -> Generator[Detections, None, None]:
        for img_path, (scores, bboxes, landmarks) in zip(img_paths, detector(img_paths)):
            if len(landmarks) < 1:
                img = cv2.imread(str(img_path))
                if img is None:
                    yield [], [], []
                    continue
                dets, landmarks = backup.detect(img)
                scores = dets[:, -1]
                bboxes = dets[:, :4]
            yield scores, bboxes, landmarks

    return detect


def pipeline_detector(img_paths, small_face: int = -1):
    detector = get_detector_with_backup()
    prep_imgs = {}

    transform_fn = transforms.Compose([
        transforms.Resize(128, keep_ratio=True),
        transforms.CenterCrop(112),
    ])

    def choose_center(img):
        return transform_fn(mx.nd.array(img[:, :, ::-1])).asnumpy()

    print('Preparing detector')
    for img_path, (scores, bboxes, landmarks) in tqdm(zip(img_paths, detector(img_paths))):
        img = cv2.imread(str(img_path))
        height, width = img.shape[:2]
        if len(landmarks) < 1:
            # logging.warning(f'smth wrong with {img_path}')
            prep_imgs[img_path] = choose_center(img)
            continue
        face_idx = choose_center_face(scores, bboxes, width, height)
        if small_face > 0:
            bbox = bboxes[face_idx]
            if bbox[2] - bbox[0] < small_face or bbox[3] - bbox[1] < small_face:
                # logging.warning(f'small face in image {img_path}')
                prep_imgs[img_path] = choose_center(img)
                continue
        warped_img = norm_crop(img, landmarks[face_idx])
        prep_imgs[img_path] = warped_img[:, :, ::-1]

    def detect(img_path: Path):
        return prep_imgs[img_path]

    return detect


def mxnet_feature_extractor(
        model_name: str = 'arcface_r100_v1',
        epoch_num: int = 0,
        use_flip: bool = False,
        ctx: mx.Context = mx.cpu()):
    embeddings = dict()
    if model_name == 'arcface_r100_v1':
        model = model_zoo.get_model(model_name)
        if ctx.device_type.startswith('cpu'):
            ctx_id = -1
        else:
            ctx_id = ctx.device_id
        model.prepare(ctx_id=ctx_id)
        model = model.model
    else:
        sym, arg_params, aux_params = mx.model.load_checkpoint(model_name, epoch_num)
        sym = sym.get_internals()['fc1_output']
        model = mx.mod.Module(symbol=sym, context=ctx, label_names=None)
        data_shape = (2 if use_flip else 1, 3, 112, 112)
        model.bind(data_shapes=[('data', data_shape)], for_training=False)
        model.set_params(arg_params, aux_params)
        # warmup
        data = mx.nd.zeros(shape=data_shape)
        db = mx.io.DataBatch(data=(data,))
        model.forward(db, is_train=False)
        embedding = model.get_outputs()[0].asnumpy()
        model = model

    hashed_imgs = {}
    hash_fn = xxhash.xxh64()

    def extract(prep_img):
        hash_fn.update(prep_img)
        img_hash = hash_fn.hexdigest()
        hash_fn.reset()
        if img_hash not in hashed_imgs:
            prep_img2 = prep_img.transpose((2, 0, 1)).expand_dims(0).astype(np.float32)
            if use_flip:
                flipped = mx.nd.flip(prep_img, axis=1).transpose((2, 0, 1)).expand_dims(0).astype(np.float32)
                img = mx.nd.concatenate([prep_img2, flipped])
            else:
                img = prep_img2
            batch = mx.io.DataBatch([img])
            model.forward(batch, is_train=False)
            hashed_imgs[img_hash] = model.get_outputs().asnumpy().mean(axis=0)
        return hashed_imgs[img_hash]

    return extract
