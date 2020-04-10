import numpy as np
from sklearn import neighbors
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from classifier.evaluation import get_accuracy, get_f1, get_recall, get_precision


def knn_classifier(trainX, trainY, n_neighbours):
    clf = neighbors.KNeighborsClassifier(n_neighbours)
    clf.fit(trainX, trainY)
    return clf


def knn_predict(clf, dataX):
    return clf.predict(dataX)


def knn_kfold_fold_search(dataX, dataY):
    n_accs = []
    n_precs = []
    n_recs = []
    n_f1s = []
    n_folds = range(2, 15)
    for i in n_folds:
        kf = KFold(n_splits=i)
        kf.get_n_splits(dataX)
        knn_accs = np.array([])
        knn_precs = np.array([])
        knn_recs = np.array([])
        knn_f1s = np.array([])
        for train_index, test_index in kf.split(dataX):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = dataX[train_index], dataX[test_index]
            y_train, y_test = dataY[train_index], dataY[test_index]
            knn = knn_classifier(X_train, y_train, 3)
            knn_pred = knn_predict(knn, X_test)
            knn_prec = get_precision(y_test, knn_pred)
            knn_precs = np.append(knn_precs, knn_prec)
            knn_recall = get_recall(y_test, knn_pred)
            knn_recs = np.append(knn_recs, knn_recall)
            knn_f1 = get_f1(y_test, knn_pred)
            knn_f1s = np.append(knn_f1s, knn_f1)
            knn_acc = get_accuracy(y_test, knn_pred)
            knn_accs = np.append(knn_accs, knn_acc)

        n_accs.append(knn_accs)
        n_precs.append(knn_precs)
        n_recs.append(knn_recs)
        n_f1s.append(knn_f1s)

    acc_avg = [np.mean(x) for x in n_accs]
    acc_var = [np.var(x) for x in n_accs]
    prec_avg = [np.mean(x) for x in n_precs]
    prec_var = [np.var(x) for x in n_precs]
    rec_avg = [np.mean(x) for x in n_recs]
    rec_var = [np.var(x) for x in n_recs]
    f1_avg = [np.mean(x) for x in n_f1s]
    f1_var = [np.var(x) for x in n_f1s]
    print(acc_avg)
    print(acc_var)
    fig = plt.figure()
    ax = fig.gca()
    plt.plot(n_folds, acc_avg, label='Mean Accuracy')
    plt.plot(n_folds, prec_avg, label='Mean Precision')
    plt.plot(n_folds, rec_avg, label='Mean Recall')
    plt.plot(n_folds, f1_avg, label='Mean F1-score')
    plt.xlabel('# of folds')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.show()
    fig.savefig('fold_mean.png', dpi=600)

    fig_var = plt.figure()
    ax_var = fig_var.gca()
    plt.plot(n_folds, acc_var, label='Accuracy Variance')
    plt.plot(n_folds, prec_var, label='Precision Variance')
    plt.plot(n_folds, rec_var, label='Mean Recall Variance')
    plt.plot(n_folds, f1_var, label='Mean F1-score Variance')
    plt.xlabel('# of folds')
    ax_var.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend()
    plt.show()
    fig_var.savefig('fold_var.png', dpi=600)


def knn_search(dataX, dataY, n_folds=7):
    n_neighbours = range(1, 20)
    n_acc = []
    n_prec = []
    n_rec = []
    n_f1 = []
    for i in n_neighbours:

        kf = KFold(n_splits=n_folds)
        kf.get_n_splits(dataX)
        knn_accs = []
        knn_precs = []
        knn_recs = []
        knn_f1s = []
        for train_index, test_index in kf.split(dataX):
            # print("TRAIN:", train_index, "TEST:", test_index)
            X_train, X_test = dataX[train_index], dataX[test_index]
            y_train, y_test = dataY[train_index], dataY[test_index]
            knn = knn_classifier(X_train, y_train, i)
            knn_pred = knn_predict(knn, X_test)
            knn_prec = get_precision(y_test, knn_pred)
            knn_precs.append(knn_prec)
            knn_recall = get_recall(y_test, knn_pred)
            knn_recs.append(knn_recall)
            knn_f1 = get_f1(y_test, knn_pred)
            knn_f1s.append(knn_f1)
            knn_acc = get_accuracy(y_test, knn_pred)
            knn_accs.append(knn_acc)
        print(
            "N:{}\tAcc:{}\tRecall:{}\tPrecision:{}\tF1-score:{}".format(i, np.mean(np.array(knn_accs)), np.mean(np.array(knn_recs)),
                                                                  np.mean(np.array(knn_precs)), np.mean(np.array(knn_f1s))))
        n_acc.append(knn_accs)
        n_prec.append(knn_precs)
        n_rec.append(knn_recs)
        n_f1.append(knn_f1s)

    np_acc = np.array(n_acc)
    np_prec = np.array(n_prec)
    np_rec = np.array(n_rec)
    np_f1 = np.array(n_f1)

    accs = np.mean(np_acc, axis=1)
    precs = np.mean(np_prec, axis=1)
    recs = np.mean(np_rec, axis=1)
    f1s = np.mean(np_f1, axis=1)

    fig = plt.figure()
    ax = fig.gca()
    plt.plot(n_neighbours, accs, label='Accuracy')
    plt.plot(n_neighbours, precs, label='Precision')
    plt.plot(n_neighbours, recs, label='Recall')
    plt.plot(n_neighbours, f1s, label='F1-score')
    plt.xlabel('k (# of neighbours)')
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    plt.legend(loc=5)
    plt.show()
    fig.savefig('knn_n_neighbours_search.png', dpi=600)
