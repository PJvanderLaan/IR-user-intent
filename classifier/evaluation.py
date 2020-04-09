from sklearn.metrics import multilabel_confusion_matrix


def get_accuracy(testY, predY):
    accuracies = []
    for label_index, singleY in enumerate(testY):
        correct = 0
        union = 0
        for i, value in enumerate(singleY):
            if value == True and predY[label_index][i] == True:
                correct = correct + 1

            if value == True or predY[label_index][i] == True:
                union = union + 1
        accuracies.append(correct/union)

    return sum(accuracies)/len(accuracies)


def get_f1(testY, predY):
    precision = get_precision(testY, predY)
    recall = get_recall(testY, predY)

    return 2 * ((precision * recall)/(precision + recall))


def get_precision(testY, predY):
    precisions = []
    for label_index, singleY in enumerate(testY):
        correct = 0
        predicted = 0
        for i, value in enumerate(singleY):
            if value == True and predY[label_index][i] == True:
                correct = correct + 1

            if predY[label_index][i] == True:
                predicted = predicted + 1
        if (predicted != 0):
            precisions.append(correct/predicted)
        # else:
        #     precisions.append(0)

    return sum(precisions)/len(precisions)


def get_recall(testY, predY):
    precisions = []
    for label_index, singleY in enumerate(testY):
        correct = 0
        true_labels = 0
        for i, value in enumerate(singleY):
            if value == True and predY[label_index][i] == True:
                correct = correct + 1

            if testY[label_index][i] == True:
                true_labels = true_labels + 1

        precisions.append(correct/true_labels)

    return sum(precisions)/len(precisions)
