import functools as ft
from collections import defaultdict

import numpy as np

from .utils import bounding_box_iou, cast_recarray
from .retrieval import interpolated_average_precision_score


__all__ = ['map_score_evaluator', 'match_detections', 'merge_rank_arrays']


BBOX_DTYPE = [('x1', np.float_), ('y1', np.float_), ('x2', np.float_),
              ('y2', np.float_)]


def _match_single_class(gt, dt, iou_thresh=0.5):
    gt = np.array(gt, dtype=BBOX_DTYPE)
    dt = np.asarray(dt, dtype=BBOX_DTYPE + [('score', np.float_)])

    if len(gt) == 0:
        return [(0, score) for score in dt['score']]

    gt_seen = np.zeros(len(gt), dtype=bool)
    bboxes_gt = gt.view((np.float_, len(gt.dtype.names)))

    # mergesort since its stable
    dt = dt[np.argsort(dt['score'], kind='mergesort')[::-1]]
    result = []

    for i, (*bbox_pred, score_pred) in enumerate(dt):
        ious, = bounding_box_iou(np.array(bbox_pred)[None], bboxes_gt)
        idx = np.argmax(ious)

        if (ious[idx] >= iou_thresh) and (not gt_seen[idx]):
            result.append((1, score_pred))
            gt_seen[idx] = True
        else:
            result.append((0, score_pred))

    result += [(1.0, 0)] * (len(gt_seen) - np.sum(gt_seen))
    return result


def match_detections(gt, dt, *, iou_thresh=0.5):
    gt = cast_recarray(gt, dtype=BBOX_DTYPE + [('class', np.int_)])
    dt = cast_recarray(dt, dtype=BBOX_DTYPE + [('class', np.int_), ('score', np.float_)])

    result = dict()
    for c in set(gt['class']).union(dt['class']):
        gt_c = gt[gt['class'] == c][['x1', 'y1', 'x2', 'y2']]
        dt_c = dt[dt['class'] == c][['x1', 'y1', 'x2', 'y2', 'score']]
        result[c] = _match_single_class(gt_c, dt_c, iou_thresh=iou_thresh)

    return result


RANK_ARRAY_DTYPE = [('gt', np.uint8), ('dt', np.float_)]

def merge_rank_arrays(rank_arrays):
    result = defaultdict(list)

    for arrays in rank_arrays:
        for key, y in arrays.items():
            result[key] += y

    return {key: np.array(val, dtype=RANK_ARRAY_DTYPE)
            for key, val in result.items() if len(val) > 0}


def map_score_evaluator(groundtruths, detections, *, iou_thresh=0.5,
                        max_detections=None, weighted=False,
                        ap_fun=interpolated_average_precision_score):
    rank_arrays = (
        match_detections(gt, dt[:max_detections], iou_thresh=iou_thresh)
        for gt, dt in zip(groundtruths, detections))
    rank_arrays = merge_rank_arrays(rank_arrays)

    if weighted:
        rank_arrays = np.concatenate(list(rank_arrays.values()))
        return ap_fun(rank_arrays['gt'], rank_arrays['dt'])
    else:
        ap_scores = np.array(
            [ap_fun(y['gt'], y['dt']) for y in rank_arrays.values()])
        return np.mean(ap_scores[~np.isnan(ap_scores)])
