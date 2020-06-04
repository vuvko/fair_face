# ECCV 2020 ChaLearn Looking at People Fair Face Recognition challenge

## Data

### Preparing

To download the data for training please use [challenge's page](https://competitions.codalab.org/competitions/24184#participate).

For face extraction you'll need face detection models, you can find them [here](https://yadi.sk/d/ppwtvSvyqZ_iXg).
Put them in `~/models` or provide path for `get_retina_det` function.

After that you need to run `prepare.py` script (see it's arguments) to extract aligned faces.

Use `data/train_df.csv` for training and `data/val_df.csv` for validating fitted model.
We'll need cross-validation pipeline later.

### External data

## Validation


## Training


## Submitting

For now only baseline solution is implemented.
See `submit_arcface.py` for the reference.
