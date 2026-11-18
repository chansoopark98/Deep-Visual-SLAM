"""dvslam: build only `cuda_ba_metric` (BA + log-depth prior unary, plan v3 item 1.1).

Same sources as `cuda_ba`, under a different module name, so the frozen `cuda_ba` build used by
dvslam/odometry/dpvo_mono.py is never overwritten (other processes may have it loaded).

  MAX_JOBS=8 python setup_metric.py build_ext --inplace
"""
import os
import os.path as osp

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

ROOT = osp.dirname(osp.abspath(__file__))
os.environ.setdefault('TORCH_CUDA_ARCH_LIST', '12.0')
EIGEN = osp.join(ROOT, 'thirdparty/eigen-3.4.0')
if not osp.isdir(EIGEN):
    EIGEN = os.environ.get('EIGEN_INCLUDE_DIR', '/usr/include/eigen3')

setup(
    name='dpvo_metric_ext',
    ext_modules=[
        CUDAExtension('cuda_ba_metric',
                      sources=['dpvo/fastba/ba.cpp', 'dpvo/fastba/ba_cuda.cu', 'dpvo/fastba/block_e.cu'],
                      extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']},
                      include_dirs=[EIGEN]),
    ],
    cmdclass={'build_ext': BuildExtension})
