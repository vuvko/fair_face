from pathlib import Path
import argparse
import pandas as pd
import cv2
import albumentations as alb
import numpy as np
import mxnet as mx
import lib
import cv2
import albumentations as alb
from lib.data import load_info
from lib.basic_config import BasicConfig


def config_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser('Training')
    parser.add_argument('data_path', type=str,
                        help='Path to the folder with images')
    return parser


def run_train(data_path: Path):
    cfg = BasicConfig(
        seed=444,
        name='ultimate2',
        num_workers=6,
        gpus=(0,),
        batch_size=32,
        num_epochs=10,
        steps=(3, 5, np.inf),
        train_augmentations=alb.Compose([
            alb.OneOf([
                alb.MotionBlur(blur_limit=5, p=0.2),
                alb.MedianBlur(blur_limit=3, p=0.1),
                alb.Blur(blur_limit=5, p=0.1)
            ], p=0.2),
            alb.OneOf([
                alb.ImageCompression(70, compression_type=alb.ImageCompression.ImageCompressionType.JPEG),
                alb.ImageCompression(70, compression_type=alb.ImageCompression.ImageCompressionType.WEBP)
            ], p=0.2),
            alb.Resize(128, 128),
            alb.OneOf([
                alb.CLAHE(clip_limit=2),
                alb.IAASharpen(),
                alb.IAAEmboss(),
                alb.RandomBrightnessContrast(),
            ], p=0.1),
            alb.Rotate(5, border_mode=cv2.BORDER_REFLECT, p=0.2),
            alb.OneOf([
                alb.RandomResizedCrop(112, 112, scale=(0.9, 1.0), ratio=(0.8, 1.1), p=0.5),
                alb.Resize(112, 112, p=0.5),
            ], p=1.0),
            alb.HorizontalFlip(p=0.5),
            alb.HueSaturationValue(p=0.7),
            alb.ChannelShuffle(p=0.5)
        ]),
        normalize=True,
        uniform_subjects=True,
        classifier_mult=100,
        lr_factor=0.1,
        initial_lr=1e-5
    )
    np.random.seed(cfg.seed)
    mx.random.seed(cfg.seed)
    train_df = load_info(data_path, Path('data/train_df.csv'))
    lib.train.train(cfg, train_df)


if __name__ == '__main__':
    args = config_parser().parse_args()
    run_train(Path(args.data_path))
