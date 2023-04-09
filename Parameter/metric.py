import numpy as np


def matrix_change(conf_mat, num_classes=5):
    Matrix_data = conf_mat[:num_classes, :num_classes]
    return Matrix_data


def confusion_matrix(pred, label, num_classes):
    mask = (label >= 0) & (label < num_classes)
    conf_mat = np.bincount(num_classes * label[mask].astype(int) + pred[mask], minlength=num_classes ** 2).reshape(
        num_classes, num_classes)
    Matrix_data = matrix_change(conf_mat)
    return Matrix_data


def evaluate(Matrix_data):
    matrix = Matrix_data
    acc = np.diag(matrix).sum() / matrix.sum()
    acc_per_class = np.diag(matrix) / matrix.sum(axis=1)
    pre = np.nanmean(acc_per_class)

    recall_class = np.diag(matrix) / matrix.sum(axis=0)
    recall = np.nanmean(recall_class)

    F1_score = (2 * pre * recall) / (pre + recall)

    IoU = np.diag(matrix) / (matrix.sum(axis=1) + matrix.sum(axis=0) - np.diag(matrix))
    mean_IoU = np.nanmean(IoU)

    # 求kappa
    pe = np.dot(np.sum(matrix, axis=0), np.sum(matrix, axis=1)) / (matrix.sum() ** 2)
    kappa = (acc - pe) / (1 - pe)
    return acc, acc_per_class, pre, IoU, mean_IoU, kappa, F1_score, recall
