import itertools as it
import functools as ft

import numpy as np


def relu_(x):
    x[x < 0] = 0
    return x


def bounding_box_iou(bboxes1, bboxes2):
    """Bounding boxes in format xmin, ymin, xmax, ymax in absolute
    units
    """
    bboxes1 = np.array(bboxes1)
    bboxes2 = np.array(bboxes2)
    assert bboxes1.shape[-1] == bboxes2.shape[-1] == 4

    x1l, y1t, x1r, y1b = np.moveaxis(bboxes1, -1, 0)
    x2l, y2t, x2r, y2b = np.moveaxis(bboxes2, -1, 0)

    xil = np.maximum(x1l[:, None], x2l[None, :])
    yit = np.maximum(y1t[:, None], y2t[None, :])
    xir = np.minimum(x1r[:, None], x2r[None, :])
    yib = np.minimum(y1b[:, None], y2b[None, :])

    area_i = relu_(xir - xil + 1) * relu_(yib - yit + 1)
    area_1 = (x1r - x1l + 1) * (y1b - y1t + 1)
    area_2 = (x2r - x2l + 1) * (y2b - y2t + 1)
    area_u = (area_1[:, None] + area_2[None, :]) - area_i
    return relu_(area_i / area_u)


def iterbatch(iterable, batch_size=None):
    if batch_size is None:
        yield [iterable]
    else:
        iterator = iter(iterable)
        try:
            while True:
                first_elem = next(iterator)
                yield it.chain((first_elem,),
                               it.islice(iterator, batch_size - 1))
        except StopIteration:
            pass


def cast_recarray(array, dtype):
    if isinstance(array, np.recarray):
        return array.astype(dtype)
    else:
        return np.rec.fromarrays(array.T, dtype=dtype).reshape(len(array))
