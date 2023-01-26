import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import pickle

import numpy as np

model = pickle.load(open("model", "rb"))

print(model.leaderboard())