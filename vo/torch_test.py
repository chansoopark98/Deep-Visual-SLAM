import torch

print("=" * 50)
print("PyTorch GPU 정보 확인")
print("=" * 50)

# CUDA 사용 가능 여부 확인
print(f"CUDA 사용 가능: {torch.cuda.is_available()}")
print(f"cuDNN 사용 가능: {torch.backends.cudnn.enabled}")
print(f"PyTorch 버전: {torch.__version__}")

if torch.cuda.is_available():
    # GPU 개수
    gpu_count = torch.cuda.device_count()
    print(f"\n사용 가능한 GPU 개수: {gpu_count}")
    
    # 현재 GPU
    current_device = torch.cuda.current_device()
    print(f"현재 사용 중인 GPU 인덱스: {current_device}")
    
    # 각 GPU 정보 출력
    print("\n각 GPU 정보:")
    for i in range(gpu_count):
        print(f"\nGPU {i}:")
        print(f"  - 이름: {torch.cuda.get_device_name(i)}")
        print(f"  - 총 메모리: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print(f"  - 현재 할당된 메모리: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        print(f"  - 캐시된 메모리: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")
    
    # GPU 가속 테스트
    print("\n" + "=" * 50)
    print("GPU 가속 테스트")
    print("=" * 50)
    
    # CPU와 GPU 속도 비교
    import time
    
    # 행렬 크기
    size = 5000
    
    # CPU 테스트
    cpu_a = torch.randn(size, size)
    cpu_b = torch.randn(size, size)
    
    start_time = time.time()
    cpu_c = torch.matmul(cpu_a, cpu_b)
    cpu_time = time.time() - start_time
    
    print(f"CPU 행렬 곱셈 시간 ({size}x{size}): {cpu_time:.4f}초")
    
    # GPU 테스트
    gpu_a = cpu_a.cuda()
    gpu_b = cpu_b.cuda()
    
    # GPU 워밍업
    _ = torch.matmul(gpu_a, gpu_b)
    torch.cuda.synchronize()
    
    start_time = time.time()
    gpu_c = torch.matmul(gpu_a, gpu_b)
    torch.cuda.synchronize()
    gpu_time = time.time() - start_time
    
    print(f"GPU 행렬 곱셈 시간 ({size}x{size}): {gpu_time:.4f}초")
    print(f"속도 향상: {cpu_time/gpu_time:.2f}배")
    
    # 결과 검증
    print(f"\n결과 일치 여부: {torch.allclose(cpu_c, gpu_c.cpu(), rtol=1e-5)}")
    
else:
    print("\nCUDA를 사용할 수 없습니다.")
    print("GPU 드라이버와 CUDA 설치를 확인하세요.")