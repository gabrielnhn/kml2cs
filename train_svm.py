# import autosklearn.classification
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics
import pickle
# from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report


import numpy as np

if __name__ == "__main__":

    X_train = np.load("TRAIN_X.npy", allow_pickle=True)
    X_test = np.load("TEST_X.npy", allow_pickle=True)

    y_train = np.load("TRAIN_y.npy", allow_pickle=True)
    y_test = np.load("TEST_y.npy", allow_pickle=True)

    # k = int(len(X_train)**(1/2))

    svc = SVC(kernel='poly')


    print ('Fitting svc')
    svc.fit(X_train, y_train)

    print ('Predicting...')
    y_pred = svc.predict(X_test)

    print (f'Accuracy: ',  svc.score(X_test, y_test))
    print(classification_report(y_test, y_pred))
    print("\n")


# save model 
filename = 'knn'
pickle.dump(knn, open(filename, 'wb'))
