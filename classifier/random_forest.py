from sklearn.ensemble import RandomForestClassifier


def rf_classifier(trainX, trainY, n_estimators, criterion):
    clf = RandomForestClassifier(n_estimators=n_estimators, criterion=criterion)
    clf.fit(trainX, trainY)
    return clf


def rf_predict(clf, dataX):
    return clf.predict(dataX)


# # Hyper parameter optimization for random forest using grid search.
# def calculate_best_parameters(train, test):
#     clf = GridSearchCV(LabelPowerset(), parameters, scoring=get_accuracy)
#     clf.fit(train, test)
#
#     print(clf.best_params_, clf.best_score_)