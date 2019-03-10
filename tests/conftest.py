import functools as ft
import json
from collections import namedtuple
from operator import add
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def pytest_configure():
    # index, iou threshold, max_detections
    pytest.coco_params = [(1, 0.5, 100), (2, 0.75, 100)]


@pytest.fixture(scope='session')
def datadir():
    return Path(__file__).parents[1] / 'data'


def load_json_detection_data(datadir):
    def parse_bbox(entry):
        x, y, w, h = entry
        return {'x': x, 'y': y, 'w': w, 'h': h}

    with open(datadir / 'instances_val2014.json') as buf:
        annotations = json.load(buf)['annotations']
        groundtruth = [
            {'image_id': row['image_id'], 'category_id': row['category_id'],
             **parse_bbox(row['bbox'])}
            for row in annotations]
        groundtruth = pd.DataFrame.from_records(groundtruth)
        image_ids = np.sort(np.unique(groundtruth['image_id'].values))[0:100]
        groundtruth = groundtruth.loc[groundtruth['image_id'].isin(image_ids)] \
            .to_dict('records')


    with open(datadir / 'instances_val2014_fakebbox100_results.json') as buf:
        annotations = json.load(buf)
        detection = [
            {'image_id': row['image_id'], 'category_id': row['category_id'],
             'score': row['score'], **parse_bbox(row['bbox'])}
            for row in annotations]

    return {'detection': detection, 'groundtruth': groundtruth}


def to_dataframe(records):
    df = pd.DataFrame.from_records(records)
    df['x2'] = df['x'] + df['w']
    df['y2'] = df['y'] + df['h']
    df = df.rename(columns={'x': 'x1', 'y': 'y1', 'category_id': 'class'})
    return df


@pytest.fixture(scope='session')
def detection_data(datadir, request):
    result = request.config.cache.get('detection_data', None)
    if result is None:
        result = load_json_detection_data(datadir)
        request.config.cache.set('detection_data', result)

    Data = namedtuple('Data', 'gt, dt')
    groundtruth = to_dataframe(result['groundtruth'])
    detection = to_dataframe(result['detection'])
    return Data(groundtruth, detection)


@pytest.fixture(scope='session')
def coco_results(datadir, request):
    result = request.config.cache.get('coco_results', None)

    if result is None:
        gt = COCO(str(datadir / 'instances_val2014.json'))
        dt = gt.loadRes(str(datadir / 'instances_val2014_fakebbox100_results.json'))

        coco_eval = COCOeval(gt, dt, 'bbox')
        img_ids = sorted(gt.getImgIds())[0:100]
        coco_eval.params.imgIds  = img_ids
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        result = coco_eval.stats
        request.config.cache.set('coco_results', list(result))

    return result
