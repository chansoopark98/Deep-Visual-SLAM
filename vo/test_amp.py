import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
import tensorflow as tf
import keras
import numpy as np
from model.depth_net import DispNet
import yaml

def create_dummy_data(batch_size, height, width):
    """더미 데이터 생성"""
    # 입력 이미지 (노이즈 + 간단한 패턴)
    images = np.random.normal(0.5, 0.1, (batch_size, height, width, 3)).astype(np.float32)
    
    # 타겟 depth (간단한 그라디언트 패턴)
    target_depths = []
    for scale in range(4):
        h = height // (2**scale)
        w = width // (2**scale)
        depth = np.zeros((batch_size, h, w, 1), dtype=np.float32)
        for i in range(batch_size):
            # 간단한 그라디언트 패턴
            depth[i, :, :, 0] = np.linspace(0.1, 0.9, h)[:, np.newaxis]
        target_depths.append(depth)
    
    return images, target_depths

def test_depth_model_training():
    """Depth 모델만 테스트"""
    print("=== Testing Depth Model Training ===")
    
    # Configuration
    batch_size = 4
    img_height = 240
    img_width = 320
    num_epochs = 5
    
    # Mixed Precision 설정
    policy = keras.mixed_precision.Policy('mixed_float16')
    keras.mixed_precision.set_global_policy(policy)
    print(f"Mixed Precision Policy: {policy.name}")
    
    # 모델 생성
    print("\n1. Creating DispNet model...")
    depth_net = DispNet(image_shape=(img_height, img_width), 
                       batch_size=batch_size, 
                       prefix='test_disp')
    
    # 모델 빌드
    input_shape = (batch_size, img_height, img_width, 3)
    depth_net.build(input_shape)
    
    # 더미 입력으로 모델 초기화
    dummy_input = tf.random.normal(input_shape)
    _ = depth_net(dummy_input)
    print("Model built successfully!")
    
    # Optimizer 설정
    optimizer = keras.optimizers.Adam(learning_rate=1e-4)
    optimizer = keras.mixed_precision.LossScaleOptimizer(optimizer)
    
    # 간단한 L1 loss 함수
    def compute_loss(pred_disps, target_depths):
        total_loss = 0.0
        for scale in range(len(pred_disps)):
            pred = tf.cast(pred_disps[scale], tf.float32)
            target = tf.cast(target_depths[scale], tf.float32)
            loss = tf.reduce_mean(tf.abs(pred - target))
            total_loss += loss
        return total_loss / len(pred_disps)
    
    # Training step (JIT 없음)
    @tf.function
    def train_step(images, target_depths):
        with tf.GradientTape() as tape:
            # Forward pass
            pred_disps = depth_net(images, training=True)
            
            # Loss 계산
            loss = compute_loss(pred_disps, target_depths)
            scaled_loss = optimizer.scale_loss(loss)
        
        # Gradient 계산 및 적용
        grads = tape.gradient(scaled_loss, depth_net.trainable_variables)
        optimizer.apply_gradients(zip(grads, depth_net.trainable_variables))
        
        return loss, pred_disps
    
    # Training step with JIT
    @tf.function(jit_compile=True)
    def train_step_jit(images, target_depths):
        with tf.GradientTape() as tape:
            # Forward pass
            pred_disps = depth_net(images, training=True)
            
            # Loss 계산
            loss = compute_loss(pred_disps, target_depths)
            scaled_loss = optimizer.scale_loss(loss)
        
        # Gradient 계산 및 적용
        grads = tape.gradient(scaled_loss, depth_net.trainable_variables)
        optimizer.apply_gradients(zip(grads, depth_net.trainable_variables))
        
        return loss, pred_disps
    
    # 테스트 1: JIT 없이 학습
    print("\n2. Testing without JIT compilation...")
    try:
        for epoch in range(3):
            images, target_depths = create_dummy_data(batch_size, img_height, img_width)
            images = tf.constant(images)
            target_depths = [tf.constant(td) for td in target_depths]
            
            loss, _ = train_step(images, target_depths)
            print(f"Epoch {epoch+1}, Loss: {loss.numpy():.6f}")
        print("✓ Training without JIT: SUCCESS")
    except Exception as e:
        print(f"✗ Training without JIT: FAILED")
        print(f"Error: {e}")
    
    # 테스트 2: JIT 포함 학습
    print("\n3. Testing with JIT compilation...")
    try:
        for epoch in range(3):
            images, target_depths = create_dummy_data(batch_size, img_height, img_width)
            images = tf.constant(images)
            target_depths = [tf.constant(td) for td in target_depths]
            
            loss, _ = train_step_jit(images, target_depths)
            print(f"Epoch {epoch+1}, Loss: {loss.numpy():.6f}")
        print("✓ Training with JIT: SUCCESS")
    except Exception as e:
        print(f"✗ Training with JIT: FAILED")
        print(f"Error: {e}")
    
    # 테스트 3: Mixed Precision 비활성화 후 테스트
    print("\n4. Testing with float32 only...")
    keras.mixed_precision.set_global_policy('float32')
    
    # 새 모델과 optimizer 생성
    depth_net_fp32 = DispNet(image_shape=(img_height, img_width), 
                            batch_size=batch_size, 
                            prefix='test_disp_fp32')
    depth_net_fp32.build(input_shape)
    _ = depth_net_fp32(dummy_input)
    
    optimizer_fp32 = keras.optimizers.Adam(learning_rate=1e-4)
    
    @tf.function(jit_compile=True)
    def train_step_fp32(images, target_depths):
        with tf.GradientTape() as tape:
            pred_disps = depth_net_fp32(images, training=True)
            loss = compute_loss(pred_disps, target_depths)
        
        grads = tape.gradient(loss, depth_net_fp32.trainable_variables)
        optimizer_fp32.apply_gradients(zip(grads, depth_net_fp32.trainable_variables))
        
        return loss, pred_disps
    
    try:
        for epoch in range(3):
            images, target_depths = create_dummy_data(batch_size, img_height, img_width)
            images = tf.constant(images)
            target_depths = [tf.constant(td) for td in target_depths]
            
            loss, _ = train_step_fp32(images, target_depths)
            print(f"Epoch {epoch+1}, Loss: {loss.numpy():.6f}")
        print("✓ Training with float32 + JIT: SUCCESS")
    except Exception as e:
        print(f"✗ Training with float32 + JIT: FAILED")
        print(f"Error: {e}")
    
    # 결과 요약
    print("\n=== Summary ===")
    print("This test helps identify if the issue is:")
    print("1. Mixed Precision specific")
    print("2. JIT compilation specific")
    print("3. Model architecture specific")
    print("4. Loss calculation specific")

if __name__ == '__main__':
    # GPU 설정
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"Using GPU: {gpus[0]}")
        except RuntimeError as e:
            print(e)
    
    # 테스트 실행
    test_depth_model_training()