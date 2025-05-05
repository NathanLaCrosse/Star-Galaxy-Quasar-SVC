# Star-Galaxy-Quasar-SVC
Implements a support vector classifier from sklearn to identify whether an object is a star, galaxy or quasar.

Uses data from the Sloan Digital Sky Survey, found here: https://www.kaggle.com/datasets/fedesoriano/stellar-classification-dataset-sdss17

This dataset consists of celestial objects and their light properties, which is used to classify a given source as either a star, galaxy or quasar. In this repository, I use a radial basis function support vector classifier to classify the data. Due to the size of the dataset (100,000 entries), a subset of it is taken to reduce computation time. On average, the SVC obtains ~97% accuracy. There is some variance due to the random sampling of data.

Here is the resulting confusion matrix for the SVC after a train test split with 8000 training data points and 2000 testing ones.
![SVC Matrix](https://github.com/user-attachments/assets/f32aff10-b8bf-4987-a166-56c18b9a5a9e)
