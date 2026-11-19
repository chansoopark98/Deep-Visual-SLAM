"""EXP-008: GPU time of torchvision RAFT (large / small) for one frame pair, forward + backward flow as a batch of 2.

  CUDA_VISIBLE_DEVICES=<g> python scripts/exp008_flow_timing.py
Inputs 512x512 (and resized), CUDA events, 10 warm-up + 50 timed calls, FP32 and bfloat16 autocast. Server GPU only
(RTX 5090): not an Orin number.
"""
import torch
from torchvision.models import optical_flow as of


def main():
    nets = {'large': of.raft_large(weights=of.Raft_Large_Weights.DEFAULT).eval().cuda(),
            'small': of.raft_small(weights=of.Raft_Small_Weights.DEFAULT).eval().cuda()}
    for name, net in nets.items():
        for size in (512, 256):
            for iters in (12, 20):
                for amp in (False, True):
                    x = torch.rand(2, 3, size, size, device='cuda') * 2 - 1
                    with torch.no_grad(), torch.autocast('cuda', torch.bfloat16, enabled=amp):
                        for _ in range(10):
                            net(x, x.flip(0), num_flow_updates=iters)
                        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        torch.cuda.synchronize()
                        a.record()
                        for _ in range(50):
                            net(x, x.flip(0), num_flow_updates=iters)
                        b.record()
                        torch.cuda.synchronize()
                    print(f'raft_{name} {size}x{size} iters {iters} {"bf16" if amp else "fp32"}: '
                          f'{a.elapsed_time(b) / 50:.1f} ms per pair (fwd + bwd)', flush=True)


if __name__ == '__main__':
    main()
