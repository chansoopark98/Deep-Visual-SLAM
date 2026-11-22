"""Depth priors of the tracker (DepthPrior protocol: `predict(frame) -> (depth_m HxW float32, conf HxW | None)`).

ssl.py: the self-supervised metric depth network (dvslam/ssl). perturbed.py: ground-truth depth with a scale bias and
log-normal jitter (tests and controlled experiments).
"""
