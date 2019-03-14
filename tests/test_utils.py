import pytest
import numpy as np

from debobo.utils import cast_recarray


@pytest.mark.parametrize('n', [0, 1 ,10])
def test_cast_recarray_with_array(n):
    x = np.random.randn(n, 2)
    dtype = [('x', np.float), ('y', np.float)]
    x_ = cast_recarray(x, dtype)

    assert len(x_) == n
    assert x_.dtype == dtype
    assert x_['x'].shape == (n,)
    assert x_['y'].shape == (n,)


@pytest.mark.parametrize('n', [0, 1 ,10])
def test_cast_recarray_with_recarray(n):
    dtype = [('x', np.float), ('y', np.float)]
    x = np.rec.fromarrays(np.random.randn(2, n), dtype)
    x_ = cast_recarray(x, dtype)

    assert len(x_) == n
    assert x_.dtype == dtype
    assert x_['x'].shape == (n,)
    assert x_['y'].shape == (n,)
