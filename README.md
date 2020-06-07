# ECCV 2020 ChaLearn Looking at People Fair Face Recognition challenge

## Data

### Preparing

To download the data for training please use [challenge's page](https://competitions.codalab.org/competitions/24184#participate).

For face extraction you'll need face detection models, you can find them [here](https://yadi.sk/d/ppwtvSvyqZ_iXg).
Put them in `~/models` or provide path for `get_retina_det` function.

After that, you need to run `prepare.py` script (see it's arguments) to extract aligned faces.

To check whether everything is ok, you can run `compare_arcface.py` to plot the ROC curve for the baseline model.
It should look like `arcface_roc_.png`.

Use `data/train_df.csv` for training and `data/val_df.csv` for validating the fitted model.
We'll need a cross-validation pipeline later.

### External data

## Overall structure

You can see [FITW2020](https://github.com/vuvko/fitw2020) submission for the reference.

The main idea is to provide a `Comparator` type object into the `submit` function (see `submit_arface.py` for the reference).
Baseline `Comparator` is `CompareModel` class that is searching for the saved mxnet model (finetuned ArcFace) and using it's embedding for metric computation (see `metrics.py`).

All hyperparameters are intended to be in the `BasicConfig` object.

## Validation

See `compare_arcface.py` for the reference.

For validation, you can use the `validate` function inside the `validate.py` module.
You'll need to pass a comparator you're testing and a path to the validation data.
The function returns a pair `(Labels, Predictions)` for your comparator that you can use with different metrics to score your model.

One scoring method is the AUC ROC score with plotted ROC curve.
It is implemented in `plot_roc` function inside `validate.py` module.

## Training

See `train.py` for the reference

For training, you need to provide a `BasicConfig` object with hyperparameters for the run and pandas.DataFrame with training info (including paths to training images) to the `train` function.
It will create a folder for the experiment where you can find all information to recreate and score it (pickled config, model's weights, training log).

For reproducibility, you cannot overwrite an already created experiment (that has the same name).

## Submitting

See `submit_arcface.py` for the reference.

You need to provide your comparator (similar to the validation step) for `submit` function with a path to the prepared data and CSV with validation pairs.
For reproducibility, you cannot overwrite already created submission (that has the same name).

## Literature reference

* Jiankang Deng, Jia Guo, Xue Niannan, and Stefanos Zafeiriou. Arc-face: Additive angular margin loss for deep face recognition. In CVPR, 2019.
* Brianna Maze, Jocelyn Adams, James A Duncan, Nathan Kalka, Tim Miller, Charles Otto, Anil K Jain, W Tyler Niggel, Janet Anderson, Jordan Cheney, et al. Iarpa janus benchmark-c: Face dataset and protocol. In 2018 International Conference on Biometrics (ICB), pages
* Patrick Grother, Mei Ngan, and Kayee Hanaoka. Ongoing face recognition vendor test (FRVT) part 2: Identification. Technical report, November 2018.
* Andrew Zhai and Hao-Yu Wu. Classification is a strong baseline for deep metric learning. In British Machine Vision Conference (BMVC), 2019.
