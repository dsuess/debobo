# pylint: disable=bad-continuation,no-else-return

from collections import defaultdict

import numpy as np
from typing import (Callable, DefaultDict, Dict, Hashable, Iterable, List,
                    Optional, Tuple)

from .utils import bounding_box_iou, cast_recarray

__all__ = ['match_detections', 'merge_rank_arrays', 'evaluate_frames']


BBOX_DTYPE = [('x1', np.float_), ('y1', np.float_), ('x2', np.float_),
              ('y2', np.float_)]
_ClassificationEvaluation = List[Tuple[int, float]]


def match_single_class(
    gt: np.ndarray, dt: np.ndarray, *, iou_thresh: float = 0.5) \
    -> _ClassificationEvaluation:
    """Matches the bounding boxes of the ground truth and the detections
    according to teh IOU threshold (turning the detection into a classification
    problem).

    Note that bounding boxes should be defined in coordinates preserving the
    IOU compared to absolute coordinates, e.g. relative coordinates of images
    with non-square aspect ratio do not fullfill this condition.

    Args:
        gt (np.ndarray): Numpy array with groundtruth; castable to
            `BBOX_DTYPE`, i.e.  columns corresponding to `(x1, y1, x2, y2)`
        dt (np.ndarray): Numpy array with detections with extra column `score`
            compared to `gt`
        iou_thresh (float): IOU threshold to consider a detection as positive
            (default: 0.5)

    Returns (List[Tuple[int, float]]): Detection evaluation phrased as a
        classification/ranking problem. Returned as list of tuples `(y_gt, y_dt)`.

        The first element `y_gt` of the tuple encodes whether the corresponding
        groundtruth is positive/negative. The second element `y_dt` gives the
        returned confidence of the model (if it was detected; alternatively `0`
        is returned signaling a negative detection). Hence, the following
        cases of a binary classification problem may occur

        `(y_gt == 1) and (y_dt > 0)`: True positive with confidence `y_dt`
        `(y_gt == 0) and (y_dt > 0)`: False positive with confidence `y_dt`
        `(y_gt == 1) and (y_dt == 0)`: False negative

        Since detection problems have an infinite number of true-negatives,
        the returned list only encodes the remaining three cases.
    """
    gt = np.asarray(gt, dtype=BBOX_DTYPE)
    dt = np.asarray(dt, dtype=BBOX_DTYPE + [('score', np.float_)])

    if gt.size == 0:
        return [(0, score) for score in dt['score']]

    gt_seen = np.zeros(len(gt), dtype=bool)
    bboxes_gt = gt.view((np.float_, len(gt.dtype.names)))

    # mergesort since its stable
    dt = dt[np.argsort(dt['score'], kind='mergesort')[::-1]]
    result: List[Tuple[int, float]] = []

    for *bbox_pred, score_pred in dt:
        ious, = bounding_box_iou(np.array(bbox_pred)[None], bboxes_gt)
        idx = np.argmax(ious)

        if (ious[idx] >= iou_thresh) and (not gt_seen[idx]):
            result.append((1, score_pred))
            gt_seen[idx] = True
        else:
            result.append((0, score_pred))

    result += [(1, 0)] * (len(gt_seen) - np.sum(gt_seen))
    return result


def match_detections(
    gt: np.ndarray, dt: np.ndarray, *, iou_thresh: float = 0.5) \
    -> Dict[Hashable, _ClassificationEvaluation]:
    """Matches detections on a per-class basis; turns multi-class detection
    problem into a multi-class classification problem.

    See `match_single_class` for more details.

    Args:
        gt (np.ndarray): Numpy array with groundtruth; columns corresponding
            to `(x1, y1, x2, y2, class_id)`
        gt (np.ndarray): Numpy array with detections; columns corresponding
            to `(x1, y1, x2, y2, class_id, score)`
        iou_thresh (float): IOU threshold to consider a detection as positive
            (default: 0.5)

    Returns (Dict[str, _ClassificationEvaluation]): multi-class classifcation
        evaluation results.
    """
    gt = cast_recarray(gt, dtype=BBOX_DTYPE + [('class', np.int_)])
    dt = cast_recarray(dt, dtype=BBOX_DTYPE + [('class', np.int_), ('score', np.float_)])

    result = dict()
    for c in set(gt['class']).union(dt['class']):
        gt_c = gt[gt['class'] == c][['x1', 'y1', 'x2', 'y2']]
        dt_c = dt[dt['class'] == c][['x1', 'y1', 'x2', 'y2', 'score']]
        result[c] = match_single_class(gt_c, dt_c, iou_thresh=iou_thresh)

    return result


RANK_ARRAY_DTYPE = [('gt', np.uint8), ('dt', np.float_)]

def merge_rank_arrays(
    rank_arrays: Iterable[Dict[Hashable, _ClassificationEvaluation]]) \
    -> Dict[Hashable, _ClassificationEvaluation]:
    """Merges multiple results from `match_detections` into a single
    dictionary over classification evaluation results.

    Args:
        rank_arrays (Iterable[Dict[Hashable, _ClassificationEvaluation]]):
            multi-class classification evaluation results to be merged

    Returns (Dict[Hashable, np.recarray): Merged results; for each class,
        we return a recarray with columns `gt` and `dt`

    """
    result: DefaultDict[Hashable, _ClassificationEvaluation] = defaultdict(list)

    for arrays in rank_arrays:
        for key, y in arrays.items():
            result[key] += y

    return {key: np.array(val, dtype=RANK_ARRAY_DTYPE)
            for key, val in result.items() if len(val) > 0}


def evaluate_frames(
    groundtruths: Iterable[np.ndarray],
    detections: Iterable[np.ndarray], *,
    iou_thresh: float = 0.5,
    max_detections: Optional[int] = None) \
    -> Dict[Hashable, np.recarray]:
    """Performs evaluation on a stream of detections and groundtruths.

    See `merge_rank_arrays` for the conventions used.

    Args:
        groundtruths (Iterable[np.ndarray]): frame-by-frame groundtruths
        detections (Iterable[np.ndarray]): frame-by-frame detections; order
            should match `groundtruth`
        iou_thresh (float): IOU threshold to consider a detection as positive
            (default: 0.5)
        max_detections (Optional[int]): If not `None`, it's the maximal numbers
            of detections in descending order of confidence that should be used
            for evaluation; otherwise all detections passed are used for
            evaluation

    Returns (Dict[Hashable, _ClassificationEvaluation]): Merged results
    """
    rank_arrays = (
        match_detections(gt, dt[:max_detections], iou_thresh=iou_thresh)
        for gt, dt in zip(groundtruths, detections))
    return merge_rank_arrays(rank_arrays)
