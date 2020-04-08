import numpy as np
import mxnet as mx
from pathlib import Path
import typing as t


Scores = np.ndarray
Bboxes = np.ndarray
Landmarks = np.ndarray
Img = np.ndarray  # HxWxC image in numpy (read with cv2.imread)
RGBImg = np.ndarray
MxImg = mx.nd.NDArray  # HxWxC image in mxnet (read with mx.img.imread or converted from Img)
Embedding = mx.nd.NDArray  # 1x512 image embedding (CNN output given input image)
MxImgArray = mx.nd.NDArray  # NxCxHxW batch of input images
Labels = mx.nd.NDArray  # Nx1 float unscaled kinship relation labels
ImgOrPath = t.Union[Img, Path]
ImgPairOrPath = t.Tuple[ImgOrPath, ImgOrPath]
PairPath = t.Tuple[Path, Path]

Detections = t.Tuple[Scores, Bboxes, Landmarks]
Detector = t.Callable[[t.Sequence[Path]], t.Generator[Detections, None, None]]

Backend = t.Any
BackendEl = t.Union[mx.nd.NDArray, mx.sym.Symbol]

Comparator = t.Callable[[Path, Path], float]
