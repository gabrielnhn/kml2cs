import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

import numpy as np

if __name__ == "__main__":
    X, y = sklearn.datasets.load_digits(return_X_y=True)

    X = np.load("ALL_FILES_X.npy", allow_pickle=True)
    y = np.load("ALL_FILES_Y.npy", allow_pickle=True)

    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, random_state=42, train_size=0.8)
    automl = autosklearn.classification.AutoSklearnClassifier()
    automl.fit(X_train, y_train)


    y_hat = automl.predict(X_test)
    print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_hat))

    automl.show_models()


# save model 
filename = 'model'
pickle.dump(automl, open(filename, 'wb'))