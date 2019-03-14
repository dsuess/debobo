from pathlib import Path

import numpy as np
import pytest
import torch
from debobo.adapters.ignite import MAPScore
from debobo.utils import iterbatch


def safe_to_torch(x):
    return torch.FloatTensor(x.tolist()).view(*x.shape)


@pytest.mark.parametrize('idx, iou_thresh, max_detections', pytest.coco_params)
@pytest.mark.parametrize('batch_size', [1, 4, 32])
def test_map_score_evaluator(idx, iou_thresh, max_detections, detection_data,
                             coco_results, batch_size):
    image_ids = list(set(detection_data.dt['image_id']).union(detection_data.gt['image_id']))
    groundtruths = (
        detection_data.gt.loc[detection_data.gt['image_id'] == image_id,
                              ['x1', 'y1', 'x2', 'y2', 'class']].values
        for image_id in image_ids)
    detections = (
        detection_data.dt.loc[detection_data.dt['image_id'] == image_id,
                              ['x1', 'y1', 'x2', 'y2', 'class', 'score']].values
        for image_id in image_ids)

    metric = MAPScore(iou_thresh=iou_thresh, max_detections=max_detections)
    for gt, dt in zip(iterbatch(groundtruths, batch_size),
                       iterbatch(detections, batch_size)):
        gt, dt = list(gt), list(dt)
        gt = list(map(safe_to_torch, gt))
        dt = list(map(safe_to_torch, dt))
        metric.update((dt, gt))

    ap_score = metric.compute()
    np.testing.assert_almost_equal(ap_score, coco_results[idx], decimal=2)
