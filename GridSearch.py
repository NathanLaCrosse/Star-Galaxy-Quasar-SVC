import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV

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

# Starting point for our gamma (elasticity) variable
gam = 1 / (6 * np.var(X))
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

#Perform a grid search on the data to find our optimal parameters.
classifier = SVC(kernel='rbf', decision_function_shape='ovr')
params = {'C' : np.linspace(520.0, 530.0, 5),
          'gamma' : np.linspace(gam - gam/2, gam + gam/2, 10)}
grid_search = GridSearchCV(classifier, params, cv=5)
grid_search.fit(X_train, y_train)

# Print the results.
results = pd.DataFrame(grid_search.cv_results_)
results = results[["params", "mean_test_score", "rank_test_score"]]
print(results)
print("\n\n")
print(results[results["rank_test_score"] == 1])
