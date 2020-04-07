from sklearn import neighbors
from sklearn.metrics import multilabel_confusion_matrix, accuracy_score, precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import average_precision_score

def knn_classifier(trainX, trainY, n_neighbours):
    clf = neighbors.KNeighborsClassifier(n_neighbours)
    clf.fit(trainX, trainY)
    return clf


def knn_predict(clf, dataX):
    return clf.predict(dataX)


def knn_get_accuracy(testY, predY):
    acc = accuracy_score(testY, predY)
    return acc


def knn_get_f1(testY, predY):
    f1 = f1_score(testY, predY, average='macro')
    return f1


def knn_get_precision(testY, predY):
    prec = precision_score(testY, predY, average='macro')
    return prec


def knn_get_recall(testY, predY):
    rec = recall_score(testY, predY, average='macro')
    return rec


def knn_get_confusion_matrix(clf, testY, predY):
    return multilabel_confusion_matrix(testY, predY)
