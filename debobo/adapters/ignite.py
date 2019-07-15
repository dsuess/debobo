import numpy as np
import torch
from ignite.metrics import Metric

from debobo import (interpolated_average_precision_score, match_detections,
                    merge_rank_arrays)
from debobo.retrieval import precision_recall_curve

try:
    from debobo import visualize
except ImportError:
    pass


class FakeClassNames:

    def __getitem__(self, idx):
        return idx


class RankArrays(Metric):

    def __init__(self, iou_thresh=0.5, max_detections=None,
                 ap_fun=interpolated_average_precision_score,
                 class_names=None):
        super().__init__()
        self.rank_arrays = []
        self.iou_thresh = iou_thresh
        self.max_detections = max_detections
        self.ap_fun = ap_fun
        self.class_names = class_names if class_names is not None else FakeClassNames()

    def reset(self):
        self.rank_arrays = []

    def update(self, output):
        for dt, gt in zip(*output):
            gt = gt.to('cpu').detach().numpy()
            dt = dt.to('cpu').detach().numpy()
            rank_arrays = match_detections(gt, dt[:self.max_detections],
                                           iou_thresh=self.iou_thresh)
            self.rank_arrays.append(rank_arrays)

    def compute(self):
       rank_arrays = merge_rank_arrays(self.rank_arrays)
       return {self.class_names[key]: val for key, val in rank_arrays.items()}


class APScores(RankArrays):

    def compute(self):
        rank_arrays = super().compute()
        return {name: self.ap_fun(val['gt'], val['dt'])
                for name, val in rank_arrays.items()}


class MAPScore(APScores):

    def __init__(self, weighted=False, **kwargs):
        super().__init__(**kwargs)
        self.weighted = weighted

    def compute(self):
        if self.weighted:
            rank_arrays = merge_rank_arrays(self.rank_arrays)
            rank_arrays = np.concatenate(list(rank_arrays.values()))
            return self.ap_fun(rank_arrays['gt'], rank_arrays['dt'])

        else:
            scores = np.array(list(super().compute().values()))
            return np.mean(scores[~np.isnan(scores)])


class PRCurve(RankArrays):

    def compute(self):
        rank_arrays = super().compute()
        rank_arrays = np.concatenate(list(rank_arrays.values()))
        return precision_recall_curve(rank_arrays['gt'], rank_arrays['dt'])
