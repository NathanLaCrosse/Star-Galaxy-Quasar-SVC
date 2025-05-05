import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

df = pd.read_csv("star_classification.csv")

# This column causes troubles due to its name.
df.rename(columns={'class': 'classification'}, inplace=True)

# A lot of the columns have nothing to do with classifying the object.
# For example, there is data of where the object is in the night sky and
# different ids and dates for when it was recorded.
df = df[["classification", "u", "g", "r", "i", "z", "redshift"]]

# Get rid of rows with erroneous values.
df = df.query("u >= 0")

# Reduce our sample size.
df = df.sample(10000)

X = df.iloc[:, 1:].copy().to_numpy()
y = df.iloc[:, 0].copy().to_numpy()

# Normalize the data.
X = (X - np.average(X, axis=0)) / np.std(X, axis=0)

# Split the training and testing data.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Run an SVC tuned to the optimal hyperparameters
classifier = SVC(C=527.5, gamma=0.0952, kernel='rbf', decision_function_shape='ovr')
classifier.fit(X_train, y_train)

# Print accuracy
print(f"Training Score: {classifier.score(X_train, y_train)}")
print(f"Test Score: {classifier.score(X_test, y_test)}")

# Print out a confusion matrix to understand how well the model is performing.
cm = confusion_matrix(y_test, classifier.predict(X_test), normalize='true')
disp_cm = ConfusionMatrixDisplay(cm, display_labels=classifier.classes_)
disp_cm.plot()
plt.title("Support Vector Classifier Performance")

plt.savefig("SVC Matrix.png")

plt.show()
