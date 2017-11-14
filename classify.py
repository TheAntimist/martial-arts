# import os
# from sklearn import svm
# from sklearn.metrics.pairwise import chi2_kernel
# from sklearn.externals import joblib
# import numpy as np
# from sklearn.metrics import accuracy_score
from MartialArts import MartialArts
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics.pairwise import chi2_kernel
from sklearn.metrics import accuracy_score, mean_squared_error, confusion_matrix
from sklearn.model_selection import StratifiedKFold
import numpy as np


def perclass_acc(label, y_true, y_pred):

    total_true = y_true[y_true == label].shape[0]
    temp = y_pred[y_true == label]
    total_pred_true = temp[temp == label].shape[0]
    return total_pred_true / total_true

def predict(clf, X, y):
    y_pred = clf.predict(X)
    accuracy = accuracy_score(y, y_pred)
    cf = confusion_matrix(y, y_pred, labels=range(len(dataset.labels)))
    pc_acc = [perclass_acc(i, y, y_pred) for i in range(len(dataset.labels))]
    return accuracy, cf, pc_acc

def print_pca(pc_acc):
    for i in range(len(dataset.labels)):
        print("Accuracy of class {} : {}".format(dataset.labels[i], pc_acc[i]))


dataset = MartialArts(verbose=True, read_from_file=True)
X, y = dataset.dataset()
folds=10
accuracy = []
skf = StratifiedKFold(n_splits=folds)
for train, test in skf.split(X, y):
    n_neighbours = [1, 3, 7, 10]
    print("Starting the Classification of data using Nearest Neighour.")  # Test with Manhattan Distance as well
    local_accuracy = []
    X_train, y_train = X[train], y[train]
    Xval, yval = X[test], y[test]
    local_pc = np.array((1, 1))
    for nn in n_neighbours:
        # print("NN {}:".format(nn))
        clf = KNeighborsClassifier(nn, n_jobs=3)
        clf.fit(X_train, y_train)
        a, cf, pc = predict(clf, Xval, yval)
        print("Accuracy at nn {} : {}".format(nn, a))
        local_accuracy.append(a)
    ind = np.argmax(local_accuracy)
    print("KNN accuracy is max on the Validation set at {} with accuracy: {}".format(n_neighbours[ind], local_accuracy[ind]))
    nn = n_neighbours[ind]
    accuracy.append(local_accuracy[ind])
    clf = KNeighborsClassifier(nn, n_jobs=3)
    clf.fit(X_train, y_train)
    a, cf, pc = predict(clf, Xval, yval)
    print("Confusion matrix at nn {}:\n{}".format(nn, cf))
    print_pca(pc)

print('Mean Accuracy: {}'.format(np.mean(accuracy)))
print('Standard Deviation: {}'.format(np.std(accuracy)))

accuracy.clear()

for train, test in skf.split(X, y):
    print("Starting the Classification of data using LinearSVC.")  # Test with Manhattan Distance as well

    X_train, y_train = X[train], y[train]
    Xval, yval = X[test], y[test]

    clf = LinearSVC()
    clf.fit(X_train, y_train)
    a, cf, pc = predict(clf, Xval, yval)
    print("Accuracy : {}".format(a))
    print("Confusion Matrix :\n{}".format(cf))
    print_pca(pc)
    accuracy.append(a)

print('Mean Accuracy: {}'.format(np.mean(accuracy)))
print('Standard Deviation: {}'.format(np.std(accuracy)))

accuracy.clear()

for train, test in skf.split(X, y):
    print("Starting the Classification of data using RBF.")  # Test with Manhattan Distance as well

    X_train, y_train = X[train], y[train]
    Xval, yval = X[test], y[test]

    clf = SVC(kernel='rbf')
    clf.fit(X_train, y_train)
    a, cf, pc = predict(clf, Xval, yval)
    print("Accuracy : {}".format(a))
    print("Confusion Matrix :\n{}".format(cf))
    print_pca(pc)
    accuracy.append(a)

print('Mean Accuracy: {}'.format(np.mean(accuracy)))
print('Standard Deviation: {}'.format(np.std(accuracy)))
