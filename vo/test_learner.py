import torch
import numpy as np
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from vo.monodepth_learner_torch import MonodepthLearner
from model.depthnet import DepthNet
from model.posenet import PoseNet


def create_dummy_sample(batch_size: int = 2, height: int = 192, width: int = 640, 
                       device: str = 'cuda', mode: str = 'mono'):
    """Create dummy data for testing"""
    sample = {}
    
    if mode == 'mono':
        # Monocular data
        sample['source_left'] = torch.randn(batch_size, 3, height, width, device=device)
        sample['source_right'] = torch.randn(batch_size, 3, height, width, device=device)
        sample['target_image'] = torch.randn(batch_size, 3, height, width, device=device)
    else:
        # Stereo data
        sample['source_image'] = torch.randn(batch_size, 3, height, width, device=device)
        sample['target_image'] = torch.randn(batch_size, 3, height, width, device=device)
        # Random pose (6DoF: 3 for axis-angle, 3 for translation)
        sample['pose'] = torch.randn(batch_size, 6, device=device) * 0.1
    
    # Camera intrinsics
    fx = width
    fy = height
    cx = width / 2
    cy = height / 2
    
    intrinsics = torch.tensor([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)
    
    sample['intrinsic'] = intrinsics.unsqueeze(0).repeat(batch_size, 1, 1)
    
    return sample


def test_forward_mono():
    """Test monocular forward pass"""
    print("Testing Monocular Forward Pass...")
    
    # Configuration
    config = {
        'Train': {
            'batch_size': 2,
            'img_h': 192,
            'img_w': 640,
            'num_source': 2,
            'smoothness_ratio': 0.001,
            'auto_mask': True,
            'ssim_ratio': 0.85,
            'min_depth': 0.1,
            'max_depth': 100.0
        }
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create models
    depth_net = DepthNet(num_layers=18, pretrained=False)
    pose_net = PoseNet(num_layers=18, pretrained=False, num_input_images=2)
    
    # Create learner
    learner = MonodepthLearner(depth_net, pose_net, config, device=device)
    
    # Create dummy sample
    sample = create_dummy_sample(
        batch_size=config['Train']['batch_size'],
        height=config['Train']['img_h'],
        width=config['Train']['img_w'],
        device=device,
        mode='mono'
    )
    
    # Test training mode
    print("  Testing training mode...")
    total_loss, pixel_loss, smooth_loss, pred_depths = learner.forward_mono(sample, training=True)
    
    print(f"  Total Loss: {total_loss.item():.4f}")
    print(f"  Pixel Loss: {pixel_loss.item():.4f}")
    print(f"  Smooth Loss: {smooth_loss.item():.4f}")
    print(f"  Number of depth predictions: {len(pred_depths)}")
    
    for i, depth in enumerate(pred_depths):
        print(f"  Depth scale {i} shape: {depth.shape}")
        print(f"    Min depth: {depth.min().item():.4f}, Max depth: {depth.max().item():.4f}")
    
    # Test eval mode
    print("\n  Testing evaluation mode...")
    with torch.no_grad():
        total_loss_eval, _, _, _ = learner.forward_mono(sample, training=False)
    print(f"  Eval Total Loss: {total_loss_eval.item():.4f}")
    
    # Check backward pass
    print("\n  Testing backward pass...")
    total_loss.backward()
    
    # Check gradients
    has_grad_depth = any(p.grad is not None and p.grad.abs().sum() > 0 
                         for p in depth_net.parameters() if p.requires_grad)
    has_grad_pose = any(p.grad is not None and p.grad.abs().sum() > 0 
                        for p in pose_net.parameters() if p.requires_grad)
    
    print(f"  Depth network has gradients: {has_grad_depth}")
    print(f"  Pose network has gradients: {has_grad_pose}")
    
    print("✓ Monocular forward pass test completed!\n")


def test_forward_stereo():
    """Test stereo forward pass"""
    print("Testing Stereo Forward Pass...")
    
    # Configuration
    config = {
        'Train': {
            'batch_size': 2,
            'img_h': 192,
            'img_w': 640,
            'num_source': 1,
            'smoothness_ratio': 0.001,
            'auto_mask': True,
            'ssim_ratio': 0.85,
            'min_depth': 0.1,
            'max_depth': 100.0
        }
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Create models (pose_net not needed for stereo)
    depth_net = DepthNet(num_layers=18, pretrained=False)
    pose_net = PoseNet(num_layers=18, pretrained=False)  # Dummy, not used
    
    # Create learner
    learner = MonodepthLearner(depth_net, pose_net, config, device=device)
    
    # Create dummy sample
    sample = create_dummy_sample(
        batch_size=config['Train']['batch_size'],
        height=config['Train']['img_h'],
        width=config['Train']['img_w'],
        device=device,
        mode='stereo'
    )
    
    # Test forward pass
    print("  Testing training mode...")
    total_loss, pixel_loss, smooth_loss, pred_depths = learner.forward_stereo(sample, training=True)
    
    print(f"  Total Loss: {total_loss.item():.4f}")
    print(f"  Pixel Loss: {pixel_loss.item():.4f}")
    print(f"  Smooth Loss: {smooth_loss.item():.4f}")
    print(f"  Number of depth predictions: {len(pred_depths)}")
    
    for i, depth in enumerate(pred_depths):
        print(f"  Depth scale {i} shape: {depth.shape}")
    
    # Check backward pass
    print("\n  Testing backward pass...")
    total_loss.backward()
    
    has_grad = any(p.grad is not None and p.grad.abs().sum() > 0 
                   for p in depth_net.parameters() if p.requires_grad)
    print(f"  Depth network has gradients: {has_grad}")
    
    print("✓ Stereo forward pass test completed!\n")


def test_different_scales():
    """Test with different number of scales"""
    print("Testing Different Number of Scales...")
    
    for num_scales in [2, 3, 4]:
        print(f"\n  Testing with {num_scales} scales...")
        
        config = {
            'Train': {
                'batch_size': 2,
                'img_h': 192,
                'img_w': 640,
                'num_source': 2,
                'smoothness_ratio': 0.001,
                'auto_mask': True,
                'ssim_ratio': 0.85,
                'min_depth': 0.1,
                'max_depth': 100.0
            }
        }
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create models
        depth_net = DepthNet(num_layers=18, pretrained=False, scales=range(num_scales))
        pose_net = PoseNet(num_layers=18, pretrained=False)
        
        # Create learner
        learner = MonodepthLearner(depth_net, pose_net, config, device=device)
        learner.num_scales = num_scales
        
        # Re-initialize projection layers for different scales
        learner.backproject_depth = {}
        learner.project_3d = {}
        
        for scale in range(num_scales):
            h = config['Train']['img_h'] // (2 ** scale)
            w = config['Train']['img_w'] // (2 ** scale)
            
            learner.backproject_depth[scale] = BackprojectDepth(
                config['Train']['batch_size'], h, w
            ).to(device)
            
            learner.project_3d[scale] = Project3D(
                config['Train']['batch_size'], h, w
            ).to(device)
        
        # Create dummy sample
        sample = create_dummy_sample(
            batch_size=config['Train']['batch_size'],
            height=config['Train']['img_h'],
            width=config['Train']['img_w'],
            device=device,
            mode='mono'
        )
        
        # Test forward pass
        try:
            total_loss, _, _, pred_depths = learner.forward_mono(sample, training=True)
            print(f"    Loss: {total_loss.item():.4f}")
            print(f"    Depth predictions: {len(pred_depths)}")
            print("    ✓ Success!")
        except Exception as e:
            print(f"    ✗ Failed: {str(e)}")
    
    print("\n✓ Different scales test completed!\n")


def test_batch_sizes():
    """Test with different batch sizes"""
    print("Testing Different Batch Sizes...")
    
    config_template = {
        'Train': {
            'img_h': 192,
            'img_w': 640,
            'num_source': 2,
            'smoothness_ratio': 0.001,
            'auto_mask': True,
            'ssim_ratio': 0.85,
            'min_depth': 0.1,
            'max_depth': 100.0
        }
    }
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    for batch_size in [1, 2, 4]:
        print(f"\n  Testing batch size {batch_size}...")
        
        config = config_template.copy()
        config['Train']['batch_size'] = batch_size
        
        # Create models
        depth_net = DepthNet(num_layers=18, pretrained=False)
        pose_net = PoseNet(num_layers=18, pretrained=False)
        
        # Create learner
        learner = MonodepthLearner(depth_net, pose_net, config, device=device)
        
        # Create dummy sample
        sample = create_dummy_sample(
            batch_size=batch_size,
            height=config['Train']['img_h'],
            width=config['Train']['img_w'],
            device=device,
            mode='mono'
        )
        
        # Test forward pass
        try:
            total_loss, _, _, _ = learner.forward_mono(sample, training=True)
            print(f"    Loss: {total_loss.item():.4f}")
            print("    ✓ Success!")
        except Exception as e:
            print(f"    ✗ Failed: {str(e)}")
    
    print("\n✓ Batch size test completed!\n")


def test_memory_efficiency():
    """Test memory usage"""
    print("Testing Memory Efficiency...")
    
    if not torch.cuda.is_available():
        print("  CUDA not available, skipping memory test")
        return
    
    config = {
        'Train': {
            'batch_size': 4,
            'img_h': 256,
            'img_w': 832,
            'num_source': 2,
            'smoothness_ratio': 0.001,
            'auto_mask': True,
            'ssim_ratio': 0.85,
            'min_depth': 0.1,
            'max_depth': 100.0
        }
    }
    
    device = 'cuda'
    
    # Clear cache
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    
    # Initial memory
    initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB
    
    # Create models
    depth_net = DepthNet(num_layers=18, pretrained=False)
    pose_net = PoseNet(num_layers=18, pretrained=False)
    
    # Create learner
    learner = MonodepthLearner(depth_net, pose_net, config, device=device)
    
    # Memory after model creation
    model_memory = torch.cuda.memory_allocated() / 1024**2  # MB
    
    # Create dummy sample
    sample = create_dummy_sample(
        batch_size=config['Train']['batch_size'],
        height=config['Train']['img_h'],
        width=config['Train']['img_w'],
        device=device,
        mode='mono'
    )
    
    # Forward pass
    total_loss, _, _, _ = learner.forward_mono(sample, training=True)
    
    # Peak memory during forward pass
    peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
    
    print(f"  Initial memory: {initial_memory:.2f} MB")
    print(f"  Model memory: {model_memory:.2f} MB")
    print(f"  Peak memory: {peak_memory:.2f} MB")
    print(f"  Memory for forward pass: {peak_memory - model_memory:.2f} MB")
    
    print("\n✓ Memory efficiency test completed!\n")


if __name__ == "__main__":
    print("=" * 60)
    print("MonodepthLearner Test Suite")
    print("=" * 60)
    
    # Import required modules
    try:
        from model.layers import BackprojectDepth, Project3D
    except ImportError:
        print("Error: Could not import required modules.")
        print("Make sure model.layers module is available.")
        sys.exit(1)
    
    # Run tests
    test_forward_mono()
    test_forward_stereo()
    test_different_scales()
    test_batch_sizes()
    test_memory_efficiency()
    
    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)