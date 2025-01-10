import tensorflow as tf

def rescale_camera_intrinsic(original_intrinsic_matrix: tf.Tensor,
                             original_size: tuple, 
                             target_size: tuple) -> tf.Tensor:
    scale_x = tf.cast(target_size[1] / original_size[1], tf.float32)  # target_width / original_width
    scale_y = tf.cast(target_size[0] / original_size[0], tf.float32)  # target_height / original_height
    original_intrinsic_matrix = tf.tensor_scatter_nd_update(original_intrinsic_matrix, [[0, 0]], [original_intrinsic_matrix[0, 0] * scale_x])
    original_intrinsic_matrix = tf.tensor_scatter_nd_update(original_intrinsic_matrix, [[1, 1]], [original_intrinsic_matrix[1, 1] * scale_y])
    original_intrinsic_matrix = tf.tensor_scatter_nd_update(original_intrinsic_matrix, [[0, 2]], [original_intrinsic_matrix[0, 2] * scale_x])
    original_intrinsic_matrix = tf.tensor_scatter_nd_update(original_intrinsic_matrix, [[1, 2]], [original_intrinsic_matrix[1, 2] * scale_y])
    
    return original_intrinsic_matrix
