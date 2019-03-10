import pytest
from pathlib import Path
from debobo.metrics import map_score_evaluator
import numpy as np


@pytest.mark.parametrize('idx, iou_thresh, max_detections', pytest.coco_params)
def test_map_score_evaluator(idx, iou_thresh, max_detections, detection_data, coco_results):
    image_ids = list(set(detection_data.dt['image_id']).union(detection_data.gt['image_id']))
    groundtruths = (
        detection_data.gt.loc[detection_data.dt['image_id'] == image_id] \
        .to_records(index=False)[['x1', 'y1', 'x2', 'y2', 'class']]
        for image_id in image_ids)
    detections = (
        detection_data.dt.loc[detection_data.gt['image_id'] == image_id]
        .to_records(index=False)[['x1', 'y1', 'x2', 'y2', 'class', 'score']]
        for image_id in image_ids)

    y = map_score_evaluator(groundtruths, detections, iou_thresh=iou_thresh,
                            max_detections=max_detections)
    np.testing.assert_almost_equal(y, coco_results[idx])
