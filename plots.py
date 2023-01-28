import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import sklearn.model_selection
import pickle
import autosklearn.classification
import sklearn.ensemble
import math
# from sklearn.inspection import DecisionBoundaryDisplay


i = 1

# preprocess dataset, split into training and test part

X = np.load("ALL_FILES_X.npy", allow_pickle=True)
y = np.load("ALL_FILES_Y.npy", allow_pickle=True)

X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=42, train_size=0.8)

# x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
# y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5


x_min, x_max = -math.pi, math.pi
y_min, y_max = -math.pi, math.pi

# # just plot the dataset first
cm = plt.cm.RdBu
cm_bright = ListedColormap(["#FF0000", "#0000FF"])
# plt.title("Input data")

# # Plot the training points
# plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="None")
# # Plot the testing points
# plt.scatter(
#     X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6, edgecolors="None"
# )
# plt.xlim(x_min, x_max)
# plt.ylim(y_min, y_max)


# plt.xticks(())
# plt.yticks(())

# # iterate over classifiers

# ax = figure.subplot(1, 1 + 1, 2)

clf = pickle.load(open("model", "rb"))

score = clf.score(X_test, y_test)

# DecisionBoundaryDisplay.from_estimator(
#     clf, X, cmap=cm, alpha=0.8, ax=plt, eps=0.5
# )

ax = plt.subplot()
sklearn.tree.plot_tree(
    clf, X, ax=ax
)

# Plot the training points
plt.scatter(
    X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright, edgecolors="None"
)
# Plot the testing points
plt.scatter(
    X_test[:, 0],
    X_test[:, 1],
    c=y_test,
    cmap=cm_bright,
    edgecolors="k",
    alpha=0.6,
)

plt.xlim(x_min, x_max)
plt.ylim(y_min, y_max)
plt.xticks(())
plt.yticks(())
plt.title("automl")
plt.text(
    x_max - 0.3,
    y_min + 0.3,
    ("%.2f" % score).lstrip("0"),
    size=15,
    horizontalalignment="right",
)
i += 1

# figure.tight_layout()
plt.show()