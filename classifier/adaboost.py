from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.tree import DecisionTreeClassifier


def get_adaboost(trainX, trainY):
    bdt_real = AdaBoostClassifier(
        DecisionTreeClassifier(max_depth=2),
        n_estimators=600,
        learning_rate=1)
    # bdt_real = DecisionTreeClassifier(max_depth=2)
    bdt_real.fit(trainX, trainY)
    return bdt_real


def adaboost_predict(clf, testX):
    pred = clf.predict(testX)
    return pred


def adaboost_get_accuracy(testY, predY):
    acc = accuracy_score(testY, predY)
    return acc


def adaboost_get_f1(testY, predY):
    f1 = f1_score(testY, predY, average='macro')
    return f1


def adaboost_get_precision(testY, predY):
    prec = precision_score(testY, predY, average='macro')
    return prec


def adaboost_get_recall(testY, predY):
    rec = recall_score(testY, predY, average='macro')
    return rec


