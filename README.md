# ECCV 2020 ChaLearn Looking at People Fair Face Recognition challenge

## Data

### Preparing

To download the data for training please use [challenge's page](https://competitions.codalab.org/competitions/24184#participate).

For face extraction you'll need face detection models, you can find them [here](https://yadi.sk/d/ppwtvSvyqZ_iXg).
Put them in `~/models` or provide path for `get_retina_det` function.

After that you need to run `prepare.py` script (see it's arguments) to extract aligned faces.

To check whther everything is ok, you can run `compare_arcface.py` to plot ROC curve for the baseline model.
It should look like `arcface_roc_.png`.

Use `data/train_df.csv` for training and `data/val_df.csv` for validating fitted model.
We'll need cross-validation pipeline later.

### External data

## Overall structure

You can see [FITW2020](https://github.com/vuvko/fitw2020) submission for the reference.

The main idea is to provide `Comparator` type object into `submit` function (see `submit_arface.py` for the reference).
Baseline `Comparator` is `CompareModel` class that is searching for the saved mxnet model (finetued ArcFace) and using it's embedding for metric computation (see `metrics.py`).

All hyperparameters are intended to be in the `BasicConfig` object.

## Validation

See `compare_arcface.py` for the reference.

For validation you can use `validate` function inside `validate.py` module.
You'll need to pass a comparator you're testing and path to the validation data.
The function returns a pair `(Labels, Predictions)` for your comparator that you can use with different metrics to score your model.

One scoring method is AUC ROC score with plotted ROC curve.
It is implemented in `plot_roc` function inside `validate.py` module.

## Training


## Submitting

For now only baseline solution is implemented.
See `submit_arcface.py` for the reference.

## Literature reference

* Jiankang Deng, Jia Guo, Xue Niannan, and Stefanos Zafeiriou. Arc-face: Additive angular margin loss for deep face recognition. In CVPR, 2019.
* Brianna Maze, Jocelyn Adams, James A Duncan, Nathan Kalka, Tim Miller, Charles Otto, Anil K Jain, W Tyler Niggel, Janet Anderson, Jordan Cheney, et al. Iarpa janus benchmark-c: Face dataset and protocol. In 2018 International Conference on Biometrics (ICB), pages
* Patrick Grother, Mei Ngan, and Kayee Hanaoka. Ongoing facerecognition vendor test (FRVT) part 2: Identification. Technical report,November 2018.
* Andrew Zhai and Hao-Yu Wu. Classification is a strong baseline for deep metric learning. In British Machine Vision Conference (BMVC), 2019.
