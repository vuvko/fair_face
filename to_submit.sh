#!/bin/sh

python submit_experiment.py /run/media/andrey/Fast/FairFace/data_prep2/val/data /run/media/andrey/Fast/FairFace/data/val/evaluation_pairs/predictions.csv arcface_ft_norm2 10
python submit_experiment.py /run/media/andrey/Fast/FairFace/data_prep2/val/data /run/media/andrey/Fast/FairFace/data/val/evaluation_pairs/predictions.csv ultimate5 10
python submit_experiment.py /run/media/andrey/Fast/FairFace/data_prep2/val/data /run/media/andrey/Fast/FairFace/data/val/evaluation_pairs/predictions.csv ultimate7 10
