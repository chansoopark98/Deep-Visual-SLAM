"""dvslam: fixed-shape (E_max-padded) variant of DPVO's update operator (plan v3 item 1.2, §4.6).

Upstream `Update.forward` has three data-dependent pieces that force host syncs and dynamic shapes:
  * `fastba.neighbors(kk, jj)` copies the edge list to the CPU and sorts it there;
  * `SoftAgg` calls `torch.unique` (output size unknown on the host);
  * `scatter_softmax` / `scatter_sum` size their output with `int(index.max())`.
Here every edge slot of a fixed-capacity edge buffer is processed. Group ids are computed arithmetically
(patch slot modulo the patch ring, frame-pair modulo a frame ring) so all tables have a static size, and
padding (invalid) edges get a private group each, so they never mix with real edges. Neighbours come
from a dense (patch × target-frame window) table, without a sort. For real edges the result equals
upstream up to float summation order.
"""
import torch

# Frame ring for (i, j) frame-pair groups. Active edges span at most REMOVAL_WINDOW + 1 (~23) frames.
PAIR_RING = 64


def group_ids(ii, jj, kk, valid, patch_ring: int):
    """Static group ids for agg_kk (per patch) and agg_ij (per frame pair). Invalid edges: one private group each."""
    E = ii.shape[0]
    slot = torch.arange(E, device=ii.device)
    g_kk = torch.where(valid, kk % patch_ring, patch_ring + slot)
    g_ij = torch.where(valid, (ii % PAIR_RING) * PAIR_RING + (jj % PAIR_RING), PAIR_RING * PAIR_RING + slot)
    return g_kk, patch_ring + E, g_ij, PAIR_RING * PAIR_RING + E


JJ_WIN = 64          # target frames of one patch's active edges span < 2 * PATCH_LIFETIME (26) slots


def static_neighbors(g_kk, jj, valid, n_real: int, jj_base):
    """GPU version of cuda_ba.neighbors(kk, jj) without sort or host sync (CUDA-graph safe): for each edge, the
    edge of the same patch with the next smaller (ix) / larger (jx) target frame jj, -1 if none.

    A dense (patch group × JJ_WIN) table holds edge ids at column jj - jj_base (jj_base = n - JJ_WIN, a 0-d
    long tensor so a captured graph can be replayed with a new n); a running max over the columns finds the
    nearest occupied column on either side. Padding edges (valid == False) get -1."""
    E = g_kk.shape[0]
    dev = g_kk.device
    col = (jj - jj_base).clamp(0, JJ_WIN - 1)
    row = torch.where(valid, g_kk, torch.zeros_like(g_kk))
    eid = torch.arange(E, device=dev)
    flat = torch.where(valid, row * JJ_WIN + col, torch.full_like(row, n_real * JJ_WIN))   # padding -> spill slot
    table = torch.full((n_real * JJ_WIN + 1,), -1, dtype=torch.long, device=dev).scatter_(0, flat, eid)
    table = table[:-1].view(n_real, JJ_WIN)
    cols = torch.arange(JJ_WIN, device=dev).expand(n_real, JJ_WIN)
    present = table >= 0
    left = torch.where(present, cols, torch.full_like(cols, -1)).cummax(1).values           # last col <= c
    right = torch.where(present, cols, torch.full_like(cols, JJ_WIN)).flip(1).cummin(1).values.flip(1)  # first col >= c
    pc = torch.where(col > 0, left[row, (col - 1).clamp(min=0)], torch.full_like(col, -1))
    nc = torch.where(col < JJ_WIN - 1, right[row, (col + 1).clamp(max=JJ_WIN - 1)], torch.full_like(col, JJ_WIN))
    ix = torch.where(valid & (pc >= 0), table[row, pc.clamp(min=0)], torch.full_like(col, -1))
    jx = torch.where(valid & (nc < JJ_WIN), table[row, nc.clamp(max=JJ_WIN - 1)], torch.full_like(col, -1))
    return ix, jx


def _scatter_sum(src, index, size):
    out_shape = list(src.shape)
    out_shape[1] = size
    return src.new_zeros(out_shape).index_add_(1, index, src)


def static_scatter_softmax(src, index, size):
    """softmax over dim 1 within groups `index` (1-D), with a static number of groups `size`."""
    out_shape = list(src.shape)
    out_shape[1] = size
    idx = index.view(1, -1, 1).expand_as(src)
    mx = src.new_full(out_shape, float('-inf')).scatter_reduce_(1, idx, src, 'amax', include_self=True)
    e = (src - mx.gather(1, idx)).exp()
    s = _scatter_sum(e, index, size)
    return e / s.gather(1, idx)


def static_softagg(agg, x, group, size):
    w = static_scatter_softmax(agg.g(x), group, size)
    y = _scatter_sum(agg.f(x) * w, group, size)
    return agg.h(y)[:, group]


def static_update(upd, net, inp, corr, ii, jj, kk, valid, patch_ring: int, jj_base):
    """Fixed-shape equivalent of dpvo.net.Update.forward over a padded edge buffer (valid: bool (E,))."""
    net = net + inp + upd.corr(corr)
    net = upd.norm(net)

    g_kk, n_kk, g_ij, n_ij = group_ids(ii, jj, kk, valid, patch_ring)
    ix, jx = static_neighbors(g_kk, jj, valid, patch_ring, jj_base)
    mask_ix = (ix >= 0).to(net.dtype).reshape(1, -1, 1)
    mask_jx = (jx >= 0).to(net.dtype).reshape(1, -1, 1)

    net = net + upd.c1(mask_ix * net[:, ix])
    net = net + upd.c2(mask_jx * net[:, jx])

    net = net + static_softagg(upd.agg_kk, net, g_kk, n_kk)
    net = net + static_softagg(upd.agg_ij, net, g_ij, n_ij)

    net = upd.gru(net)
    return net, upd.d(net), upd.w(net)
