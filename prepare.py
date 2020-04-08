import mxnet as mx
from pathlib import Path
import logging
from lib.utils import prepare_images, choose_center_face
from lib.detect import get_retina_resnet50


if __name__ == '__main__':
    logging.basicConfig(filename='prepare.log',
                        filemode='a',
                        format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                        datefmt='%H:%M:%S',
                        level=logging.DEBUG)
    detector = get_retina_resnet50(resolution=512, ctx=mx.gpu(0), batch_size=32, num_workers=6)
    prepare_images(Path('/run/media/andrey/Fast/FairFace/data'),
                   Path('/run/media/andrey/Fast/FairFace/data_prep'),
                   detector,
                   choose_center_face)
