import numpy as np


__all__ = ['average_precision_score', 'interpolated_average_precision_score']


def _interpolate_precision(precision):
    """
    precision ordered such that corresponding recall is in descending order
    """
    return np.maximum.accumulate(precision)


def precision_recall_curve(y_true, y_score):
    order = np.argsort(y_score, kind='mergesort')[::-1]
    y_true, y_score = y_true[order], y_score[order]
    n_groundruth = np.sum(y_true)
    y_true_detected = y_true[y_score > 0]

    true_pos = np.cumsum(y_true_detected)
    false_pos = np.cumsum(1 - y_true_detected)
    recall = true_pos / n_groundruth
    precision = true_pos / (true_pos + false_pos)

    precision = np.array([0] + list(precision) + [0])
    recall = np.array([0] + list(recall) + [1])

    return precision[::-1], recall[::-1]


def _average_precision_score(y_true, y_score, interpolate=True):
    prec, rec = precision_recall_curve(y_true, y_score)
    # hard coded in pycocotools
    prec[0] = prec[-1] = 0

    if interpolate:
        prec = _interpolate_precision(prec)

    prec, rec = prec[::-1], rec[::-1]
    ii = np.ravel(np.argwhere(rec[1:] != rec[:-1]) + 1)
    return np.sum((rec[ii] - rec[ii - 1]) * prec[ii])


def average_precision_score(y_true, y_score):
    return _average_precision_score(y_true, y_score, interpolate=False)


def interpolated_average_precision_score(y_true, y_score):
    return _average_precision_score(y_true, y_score, interpolate=True)
