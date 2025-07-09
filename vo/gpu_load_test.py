#!/usr/bin/env python3
# tf_gpu_test.py
import time
import tensorflow as tf

print("✅ TensorFlow version:", tf.__version__)

# 1) GPU 감지
gpus = tf.config.list_physical_devices('GPU')
if not gpus:
    print("❌ GPU 장치가 보이지 않습니다. (tf.config.list_physical_devices('GPU') returns [])")
    exit(1)

print(f"🚀 {len(gpus)} GPU detected:")
for idx, gpu in enumerate(gpus):
    print(f"   [{idx}] {gpu}")

# 2) 간단한 행렬 곱을 CPU와 GPU에서 각각 실행해 속도 비교
M = 2048  # 행렬 크기 (원하면 수정)
with tf.device('/CPU:0'):
    a_cpu = tf.random.uniform([M, M])
    b_cpu = tf.random.uniform([M, M])
    start = time.time()
    _ = tf.matmul(a_cpu, b_cpu)
    cpu_time = time.time() - start
    print(f"🖥️  CPU matmul time: {cpu_time:.4f} s")

with tf.device('/GPU:0'):
    a_gpu = tf.random.uniform([M, M])
    b_gpu = tf.random.uniform([M, M])
    # 첫 연산은 커널 로딩 시간 때문에 느릴 수 있어 두 번 측정
    _ = tf.matmul(a_gpu, b_gpu)      # 워밍업
    start = time.time()
    _ = tf.matmul(a_gpu, b_gpu)
    gpu_time = time.time() - start
    print(f"⚡ GPU matmul time: {gpu_time:.4f} s")

speedup = cpu_time / gpu_time if gpu_time else 0
print(f"\n📊 Speed-up (CPU/GPU): {speedup:.1f}×")
