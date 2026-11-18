import torch
import cuda_ba

neighbors = cuda_ba.neighbors
reproject = cuda_ba.reproject

def BA(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, M, iterations, eff_impl=False):
    return cuda_ba.forward(poses.data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, M, t0, t1, iterations, eff_impl)


def BA_prior(poses, patches, intrinsics, target, weight, lmbda, ii, jj, kk, t0, t1, M, iterations,
             prior_disp, prior_w, prior_lambda, prior_huber=0.3, eff_impl=False):
    """dvslam: BA with a per-patch log-depth prior unary (plan v3 item 1.1), from the separate `cuda_ba_metric`
    build (setup_metric.py). prior_disp / prior_w: flat float32 (N*M,), aligned with the patch buffer."""
    import cuda_ba_metric
    return cuda_ba_metric.forward_prior(poses.data, patches, intrinsics, target, weight, lmbda, ii, jj, kk, M, t0, t1,
                                        iterations, eff_impl, prior_disp, prior_w, float(prior_lambda),
                                        float(prior_huber))
