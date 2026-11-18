"""dvslam: drop-in for the two torch_scatter ops DPVO uses, built on native torch ops.

torch_scatter has no wheel for torch 2.9 / Python 3.13 / sm_120. DPVO only calls scatter_sum and
scatter_softmax with a 1-D index along `dim`, which is all this module supports.
"""
import torch


def _index_view(index, src, dim):
    shape = [1] * src.dim()
    shape[dim] = -1
    return index.view(shape).expand_as(src)


def scatter_sum(src, index, dim=-1, dim_size=None):
    assert index.dim() == 1, 'only 1-D indices are supported'
    dim = dim % src.dim()
    if dim_size is None:
        dim_size = int(index.max()) + 1 if index.numel() > 0 else 0
    size = list(src.shape)
    size[dim] = dim_size
    return src.new_zeros(size).index_add_(dim, index, src)


def scatter_softmax(src, index, dim=-1):
    assert index.dim() == 1, 'only 1-D indices are supported'
    dim = dim % src.dim()
    n = int(index.max()) + 1 if index.numel() > 0 else 0
    size = list(src.shape)
    size[dim] = n
    idx = _index_view(index, src, dim)
    mx = src.new_full(size, float('-inf')).scatter_reduce_(dim, idx, src, 'amax', include_self=True)
    e = (src - mx.gather(dim, idx)).exp()
    s = scatter_sum(e, index, dim, n)
    return e / s.gather(dim, idx)
