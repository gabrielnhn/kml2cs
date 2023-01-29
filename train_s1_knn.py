# import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import pickle
from sklearn.neighbors import KNeighborsClassifier


import numpy as np

if __name__ == "__main__":

    X_train = np.load("TRAIN_X.npy", allow_pickle=True)
    X_test = np.load("TEST_X.npy", allow_pickle=True)

    y_train = np.load("TRAIN_y.npy", allow_pickle=True)
    y_test = np.load("TEST_y.npy", allow_pickle=True)

    knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')

    print ('Fitting knn')
    knn.fit(X_train, y_train)

    print ('Predicting...')
    y_pred = knn.predict(X_test)

    print ('Accuracy: ',  knn.score(X_test, y_test))

# save model 
filename = 'knn'
pickle.dump(knn, open(filename, 'wb'))
