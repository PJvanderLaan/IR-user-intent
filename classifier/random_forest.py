from sklearn.ensemble import ExtraTreesClassifier


def rf_classifier(trainX, trainY, n_estimators, criterion):
    clf = ExtraTreesClassifier(n_estimators=n_estimators, criterion=criterion)
    clf.fit(trainX, trainY)
    return clf


def rf_predict(clf, dataX):
    return clf.predict(dataX)
