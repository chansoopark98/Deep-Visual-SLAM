import tensorflow as tf, tf_keras

# Some helper functions
# ed = expand last dim
ed = lambda x: tf.expand_dims(x,-1)
ed2 = lambda x: ed(ed(x))
ed3 = lambda x: ed(ed2(x))

def convert_NCHW_to_NHWC(inp):
    """Convert the tensor from caffe format NCHW into tensorflow format NHWC
        
        inp: tensor 
    """
    return tf.transpose(inp,[0,2,3,1])

def convert_NHWC_to_NCHW(inp):
    """Convert the tensor from tensorflow format NHWC into caffe format NCHW 
        
        inp: tensor 
    """
    return tf.transpose(inp,[0,3,1,2])

class AddCAMCoords(tf_keras.layers.Layer): #
    """Add Camera Coord Maps to a tensor.
    
    Modified to support variable intrinsic matrices per sample.
    """
    def __init__(self, coord_maps, centered_coord, norm_coord_maps, with_r,
                 bord_dist, scale_centered_coord, fov_maps,
                 data_format='channels_last',
                 resize_policy=tf.image.ResizeMethod.BILINEAR):
        self.coord_maps = coord_maps
        self.centered_coord = centered_coord
        self.norm_coord_maps = norm_coord_maps
        self.with_r = with_r
        self.bord_dist = bord_dist
        self.scale_centered_coord = scale_centered_coord
        self.fov_maps = fov_maps
        self.data_format = data_format
        self.resize_policy = resize_policy
        super(AddCAMCoords, self).__init__()

    def additional_channels(self):
        return (self.coord_maps * 2 + 
                self.centered_coord * 2 + 
                self.norm_coord_maps * 2 + 
                self.with_r * 1 + 
                self.bord_dist * 4 + 
                self.fov_maps * 2)

    def _resize_map_(self, data, w, h):
        if self.data_format == 'channels_first':
            raise NotImplementedError('channels_first not implemented')
        else:
            data = tf.image.resize(data, [h, w], method=self.resize_policy,
                                   preserve_aspect_ratio=False, antialias=False)
            return data

    def __define_coord_channels__(self, n, x_dim, y_dim):
        """
        Returns coordinate channels for x and y:
          - x coordinates: 0 to x_dim-1
          - y coordinates: 0 to y_dim-1
        """
        xx_ones = tf.ones([n, y_dim], dtype=tf.int32)
        xx_ones = tf.expand_dims(xx_ones, -1)
        xx_range = tf.tile(tf.expand_dims(tf.range(x_dim), 0), [n, 1])
        xx_range = tf.expand_dims(xx_range, 1)
        xx_channel = tf.matmul(xx_ones, xx_range)

        yy_ones = tf.ones([n, x_dim], dtype=tf.int32)
        yy_ones = tf.expand_dims(yy_ones, 1)
        yy_range = tf.tile(tf.expand_dims(tf.range(y_dim), 0), [n, 1])
        yy_range = tf.expand_dims(yy_range, -1)
        yy_channel = tf.matmul(yy_range, yy_ones)

        if self.data_format == 'channels_last':
            xx_channel = tf.expand_dims(xx_channel, -1)
            yy_channel = tf.expand_dims(yy_channel, -1)
        else:
            xx_channel = tf.expand_dims(xx_channel, 1)
            yy_channel = tf.expand_dims(yy_channel, 1)

        xx_channel = tf.cast(xx_channel, 'float32')
        yy_channel = tf.cast(yy_channel, 'float32')
        return xx_channel, yy_channel

    def call(self, input_tensor, intrinsic):
        """
        Args:
            input_tensor: A tensor of shape (B, H, W, C) (channels_last).
            intrinsic: A tensor of shape (B, 3, 3) representing each image's intrinsic matrix.
                       Expected form:
                           [[fx, 0, cx],
                            [0, fy, cy],
                            [0,  0,  1]]
        Returns:
            A tensor with additional coordinate channels concatenated.
        """
        if self.additional_channels() == 0:
            return input_tensor

        batch_size = tf.shape(input_tensor)[0]
        height = tf.shape(input_tensor)[1]
        width = tf.shape(input_tensor)[2]

        # Extract intrinsic parameters per sample.
        fx = intrinsic[:, 0, 0]  # shape (B,)
        fy = intrinsic[:, 1, 1]
        cx = intrinsic[:, 0, 2]
        cy = intrinsic[:, 1, 2]

        # Generate coordinate channels based on the image width and height.
        xx_channel, yy_channel = self.__define_coord_channels__(batch_size, width, height)
        extra_channels = []

        # 1) Normalized coordinate maps.
        if self.norm_coord_maps:
            norm_xx_channel = (xx_channel / tf.cast(width - 1, tf.float32)) * 2.0 - 1.0
            norm_yy_channel = (yy_channel / tf.cast(height - 1, tf.float32)) * 2.0 - 1.0
            if self.with_r:
                norm_rr_channel = tf.sqrt(tf.square(norm_xx_channel - 0.5) + 
                                          tf.square(norm_yy_channel - 0.5))
                extra_channels.extend([norm_xx_channel, norm_yy_channel, norm_rr_channel])
            else:
                extra_channels.extend([norm_xx_channel, norm_yy_channel])

        # 2) Centered coordinates and Field of View (FOV) maps.
        if self.centered_coord or self.fov_maps:
            # Reshape to allow broadcasting: (B, 1, 1, 1)
            cx_exp = tf.reshape(cx, [-1, 1, 1, 1])
            cy_exp = tf.reshape(cy, [-1, 1, 1, 1])
            cent_xx_channel = xx_channel - cx_exp + 0.5
            cent_yy_channel = yy_channel - cy_exp + 0.5

            if self.fov_maps:
                fx_exp = tf.reshape(fx, [-1, 1, 1, 1])
                fy_exp = tf.reshape(fy, [-1, 1, 1, 1])
                fov_xx_channel = tf.atan(cent_xx_channel / fx_exp)
                fov_yy_channel = tf.atan(cent_yy_channel / fy_exp)
                extra_channels.extend([fov_xx_channel, fov_yy_channel])
            if self.centered_coord:
                extra_channels.extend([cent_xx_channel / self.scale_centered_coord[1],
                                       cent_yy_channel / self.scale_centered_coord[0]])

        # 3) Unnormalized coordinate maps.
        if self.coord_maps:
            extra_channels.extend([xx_channel, yy_channel])

        # Concatenate extra channels (if any) and resize them to the image dimensions.
        if extra_channels:
            extra_channels = tf.concat(extra_channels, axis=-1)
            extra_channels = self._resize_map_(extra_channels, width, height)
            extra_channels = [extra_channels]

        # 4) Distance-to-border maps.
        if self.bord_dist:
            t_xx_channel, t_yy_channel = self.__define_coord_channels__(batch_size, width, height)
            l_dist = t_xx_channel
            r_dist = tf.cast(width, tf.float32) - t_xx_channel - 1
            t_dist = t_yy_channel
            b_dist = tf.cast(height, tf.float32) - t_yy_channel - 1
            extra_channels.extend([l_dist, r_dist, t_dist, b_dist])
        # tf.stop_gradient
        # extra_channels = list(tf.stop_gradient())
        extra_channels = tf.concat(extra_channels, axis=-1)
        extra_channels = tf.stop_gradient(extra_channels)
        output_tensor = tf.concat([input_tensor, extra_channels], axis=-1)
        
        return output_tensor
    
if __name__ == '__main__':
    add_coord = AddCAMCoords(coord_maps=False,
                             centered_coord=True,
                             norm_coord_maps=True,
                             with_r=False,
                             bord_dist=False,
                             scale_centered_coord=(100, 100),
                             fov_maps=True,
                             data_format='channels_last')
    channels = add_coord.additional_channels() 
    print(channels)

    dummy_input = tf.random.normal((1,100,100,3))
    dummy_intrinsics = tf.random.normal((1,3, 3))
    dummy_image_shape = tf.TensorShape([1,100,100,3])
    outputs = add_coord.call(dummy_input, dummy_intrinsics)
    print(outputs.shape)

