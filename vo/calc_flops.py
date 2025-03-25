import tensorflow as tf, tf_keras
import numpy as np
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2

def get_flops(model, model_shape:tuple):
    """
    Calculate FLOPS for TensorFlow model
    Args:
        model: TensorFlow model
        model_shape: Input shape including batch dimension
    
    Returns:
        flops: Number of FLOPS
    """
    if not isinstance(model, tf_keras.Model):
        raise ValueError("Input model is not a keras model")
    
    # Get concrete function
    concrete = tf.function(lambda inputs: model(inputs))
    
    # Create input with batch size
    dummy_input = tf.ones(model_shape)
    concrete_func = concrete.get_concrete_function(dummy_input)
    
    # Get frozen graph
    frozen_func = convert_variables_to_constants_v2(concrete_func)
    graph_def = frozen_func.graph.as_graph_def()
    
    # Calculate FLOPS - use the frozen function's graph directly
    run_meta = tf.compat.v1.RunMetadata()
    opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
    
    # Use the frozen function's graph
    flops = tf.compat.v1.profiler.profile(graph=frozen_func.graph,
                                         run_meta=run_meta,
                                         cmd='op',
                                         options=opts)
    
    # Return FLOPS count
    if flops is not None:
        return flops.total_float_ops
    return 0

def print_model_flops(model, input_shape):
    """
    Print model FLOPS in a readable format
    Args:
        model: TensorFlow Keras model
        input_shape: Optional input shape (excluding batch dimension)
    """
    if input_shape:
        flops = get_flops(model, input_shape)
    else:
        raise ValueError("Input shape is required")
    
    
    # Convert to readable format
    if flops < 1e9:
        flops_str = f"{flops / 1e6:.2f} MFLOPs"
    else:
        flops_str = f"{flops / 1e9:.2f} GFLOPs"
    
    print(f"Model: {model.name if hasattr(model, 'name') else 'Unknown'}")
    print(f"Total FLOPs: {flops_str}")
    print(f"Params: {model.count_params():,}")
    
    return flops


if __name__ == '__main__':
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
    from model.pose_net import PoseNet
    import yaml

    # Load config
    with open('./vo/config.yaml', 'r') as f:
        config = yaml.safe_load(f)
    batch_size = 1
    image_shape = (config['Train']['img_h'], config['Train']['img_w'])
    pose_net = PoseNet(image_shape=image_shape, batch_size=batch_size, prefix='mono_posenet')
    posenet_input_shape = (batch_size, *image_shape, 6)
    pose_net.build(posenet_input_shape)

    print_model_flops(pose_net, input_shape=posenet_input_shape)