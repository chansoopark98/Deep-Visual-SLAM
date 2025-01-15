import tensorflow as tf

class FlyingChairHandler(object):
    def __init__(self, target_size: tuple) -> None:
        self.target_size = target_size        
        self.original_size = (384, 512) # FlyingChair dataset
        self.x_factor = self.target_size[1] / self.original_size[1]
        self.y_factor = self.target_size[0] / self.original_size[0]

    @tf.function(jit_compile=True)
    def preprocess(self, left, right, flow) -> tuple:
        left = tf.image.resize(left, self.target_size, method=tf.image.ResizeMethod.BILINEAR)
        right = tf.image.resize(right, self.target_size, method=tf.image.ResizeMethod.BILINEAR)
        flow = tf.image.resize(flow, self.target_size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)

        # Adjust resize factor
        rescaled_flow = flow * tf.constant([self.x_factor, self.y_factor], dtype=tf.float32)
        
        left = tf.ensure_shape(left, [*self.target_size, 3])
        right = tf.ensure_shape(right, [*self.target_size, 3])
        rescaled_flow = tf.ensure_shape(rescaled_flow, [*self.target_size, 2])
        
        return left, right, rescaled_flow

if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dummy = tf.ones((384, 512, 3))
    dummy_flow = tf.ones((384, 512, 2))
    handler = FlyingChairHandler(target_size=(792, 1408))
    a, b, c = handler.preprocess(dummy, dummy, dummy_flow)
    print(a.shape, b.shape)
    plt.imshow(a)
    plt.show()
    plt.imshow(b[:, :, 0], cmap='plasma')
    plt.show()