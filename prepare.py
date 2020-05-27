import mxnet as mx
from pathlib import Path
from insightface import model_zoo
import logging
import cv2
from lib.utils import prepare_images, choose_center_face
from lib.detect import get_retina_resnet50
from lib.mytypes import Detections, Detector
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
                    logging.warning(f'cannot load an image: {img_path}')
                    yield [], [], []
                    continue
                dets, landmarks = backup.detect(img)
                scores = dets[:, -1]
                bboxes = dets[:, :4]
            yield scores, bboxes, landmarks

    return detect


if __name__ == '__main__':
    logging.basicConfig(filename='prepare.log',
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    detector = get_detector_with_backup()
    prepare_images(Path('/run/media/andrey/Fast/FairFace/data'),
                   Path('/run/media/andrey/Fast/FairFace/data_prep2'),
                   detector,
                   choose_center_face)
