import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import pickle

import numpy as np

if __name__ == "__main__":

    X_train = np.load("TRAIN_X.npy", allow_pickle=True)
    X_test = np.load("TEST_X.npy", allow_pickle=True)

    y_train = np.load("TRAIN_y.npy", allow_pickle=True)
    y_test = np.load("TEST_y.npy", allow_pickle=True)


    automl = autosklearn.classification.AutoSklearnClassifier()
    automl.fit(X_train, y_train)

    y_hat = automl.predict(X_test)
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))

    show_modes_str=automl.show_models()
    sprint_statistics_str = automl.sprint_statistics()
    print(show_modes_str)
    print(sprint_statistics_str)

# save model 
filename = 'model'
pickle.dump(automl, open(filename, 'wb'))