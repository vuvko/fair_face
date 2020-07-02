## Infrastructure

### Training

* ~~Finetuning ArcFace on challenge's data.~~
* ~~External data preparation.~~
* ~~Abstracting from model and data (for external data training).~~
* ~~[Optuna](https://github.com/optuna/optuna) framework for hyperparameter tuning.~~
* Add instructions for training on Google Colab or Kaggle.
* Ensemble
  * add linear model on top of predictions
  ~~* aggregation~~

### Validation

* ~~Basic validation.~~
* ~~Test-time augmentation.~~
* Storing validation results in one format: table or smth like that for fast plotting and choosing.

### Submission

* Reproducibility
> Organizers strongly encourage the use of docker to facilitate reproducibility

## Methods

* ~~Basic finetuned ArcFace for classification.~~
* ~~Normalised embedding network.~~
* Different input sizes.
* ArcLoss.
* ~~Uniform sampling.~~
* ~~External additional data (VGGFace2, GLint, MS-Celeb, etc.)~~
* ~~Clusterisation based on KMeans and embeddings.~~
* ~~Clusterisation based on cutting graph (use only closest 3 face embeddings).~~
* Different aggregating methods (similar to ranking problem):
  * ~~aggregate embeddings imto single embedding,~~
  * use all embeddings and aggregate results.
