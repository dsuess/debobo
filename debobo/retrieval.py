# pylint: disable=bad-continuation,no-else-return
import functools as ft

import numpy as np
from typing import Dict, Hashable, Tuple, Callable

from .detection import _ClassificationEvaluation

__all__ = ['average_precision_score', 'interpolated_average_precision_score',
           'mean_average_precision_score']


def _interpolate_precision(precision: np.ndarray) -> np.ndarray:
    """Interpolates the precision score using `p[i] = max(p[i:])`

    Args:
        precision (np.ndarray): 1-dim array containing precision in descending
        order of corresponding recall (e.g. starting with recall=1)

    Returns (np.ndarray): Interpolated precision values
    """
    return np.maximum.accumulate(precision)


def precision_recall_curve(y_true: np.ndarray, y_score: np.ndarray) \
    -> Tuple[np.ndarray, np.ndarray]:
    """Computes the precision recall curve according to the algorithm used
    by PascalVOC.

    TODO Set precision=0 as soon as threshold == 0 (since we don't explicitely
        pass in the true negatives)

    Args:
        y_true (np.ndarray): 1-dim numpy array encoding if the groundtruth
            is positive or negative.
        y_score (np.ndarray): 1-dim numpy array containing the corresponding
            confidence values predicted by the model

    Returns (Tuple[np.ndarray, np.ndarray]): Precision and recall values in
        order of ascending recall
    """
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


def _average_precision_score(
    y_true: np.ndarray, y_score: np.ndarray, interpolate: bool = True) -> float:
    """Computes the average precision score

    Args:
        See `precision_recall_curve` for `y_true` and `y_score`
        interpolate (bool): Whether to use interpolated precision-recall
            curve or not

    Returns (float): AP score
    """
    prec, rec = precision_recall_curve(y_true, y_score)
    # hard coded in pycocotools
    prec[0] = prec[-1] = 0

    if interpolate:
        prec = _interpolate_precision(prec)

    prec, rec = prec[::-1], rec[::-1]
    ii = np.ravel(np.argwhere(rec[1:] != rec[:-1]) + 1)
    return np.sum((rec[ii] - rec[ii - 1]) * prec[ii])


average_precision_score = ft.partial(
    _average_precision_score, interpolate=False)
interpolated_average_precision_score = ft.partial(
    _average_precision_score, interpolate=True)


def mean_average_precision_score(
    rank_arrays: Dict[Hashable, np.recarray], *,
    weighted: bool = False,
    ap_fun: Callable[[np.ndarray, np.ndarray], float] = interpolated_average_precision_score) \
    -> float:
    """Computes the mean average precision score for a multi-class
    classification problem

    Args:
        rank_arrays (Dict[Hashable, _ClassificationEvaluation]): multi-class
            classification evaluation results; e.g. return value from
            `detection.evaluate_frames`
        weighted (bool): Whether to average AP scores for each class weighted
            by the number of instances of that class (default: `False`)
        ap_fun (Callable[[np.ndarray, np.ndarray], float]): Function to be
            used for computing single-class AP scores
            (default: `interpolated_average_precision_score`)

    Returns (float): mAP score
    """
    if weighted:
        rank_arrays = np.concatenate(list(rank_arrays.values()))
        return ap_fun(rank_arrays['gt'], rank_arrays['dt'])
    else:
        ap_scores = np.array(
            [ap_fun(y['gt'], y['dt']) for y in rank_arrays.values()])
        return np.mean(ap_scores[~np.isnan(ap_scores)])
