import autosklearn.classification
import autosklearn.experimental.askl2
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import pickle

import numpy as np

if __name__ == "__main__":
    X = np.load("ALL_FILES_X.npy", allow_pickle=True)
    y = np.load("ALL_FILES_Y.npy", allow_pickle=True)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=42, train_size=0.8)


    TIME_BUDGET=60
    automl = autosklearn.experimental.askl2.AutoSklearn2Classifier()
    automl.fit(X_train, y_train)

    y_hat = automl.predict(X_test)
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))

    show_modes_str=automl.show_models()
    sprint_statistics_str = automl.sprint_statistics()
    print(show_modes_str)
    print(sprint_statistics_str)


# save model 
filename = 'model_v2'
pickle.dump(automl, open(filename, 'wb'))