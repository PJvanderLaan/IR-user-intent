from sklearn import neighbors


def knn_classifier(trainX, trainY, n_neighbours):
    clf = neighbors.KNeighborsClassifier(n_neighbours)
    clf.fit(trainX, trainY)
    return clf


def knn_predict(clf, dataX):
    return clf.predict(dataX)
