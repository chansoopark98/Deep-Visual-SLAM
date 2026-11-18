"""dvslam: build `cuda_corr` and `lietorch_backends` whose kernels launch on the *current* CUDA stream (needed
for CUDA-graph capture, plan v3 item 1.2) into build/stream_ext/, leaving the frozen in-place builds untouched.
dvslam/odometry/dpvo_metric.py puts build/stream_ext first on sys.path when cuda_graph=1.

  MAX_JOBS=8 python setup_stream.py build_ext --build-lib build/stream_ext --build-temp build/stream_tmp
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
    name='dpvo_stream_ext',
    ext_modules=[
        CUDAExtension('cuda_corr',
                      sources=['dpvo/altcorr/correlation.cpp', 'dpvo/altcorr/correlation_kernel.cu'],
                      extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']}),
        CUDAExtension('lietorch_backends',
                      include_dirs=[osp.join(ROOT, 'dpvo/lietorch/include'), EIGEN],
                      sources=['dpvo/lietorch/src/lietorch.cpp', 'dpvo/lietorch/src/lietorch_gpu.cu',
                               'dpvo/lietorch/src/lietorch_cpu.cpp'],
                      extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']}),
    ],
    cmdclass={'build_ext': BuildExtension})
