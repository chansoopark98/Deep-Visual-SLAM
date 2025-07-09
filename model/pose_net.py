import tensorflow as tf
from tensorflow import keras
try:
    from .flownet import CustomFlow
    from .resnet_tf import Resnet
except:
    from flownet import CustomFlow
    from resnet_tf import Resnet


def std_conv(filter_size, out_channel, stride, use_bias=True, pad='same', name='conv'):
    conv_layer = keras.layers.Conv2D(out_channel,
                                        (filter_size, filter_size),
                                         strides=(stride, stride), 
                                         use_bias=use_bias,
                                         padding=pad,
                                         name=name+'_'+'conv')
    return conv_layer

class PoseNetAB(keras.Model):
    def __init__(self,
                 image_shape: tuple,
                 batch_size: int,
                 prefix='pose_resnet',
                 **kwargs):
        super(PoseNetAB, self).__init__(**kwargs)

        self.image_height = image_shape[0]
        self.image_width = image_shape[1]
        self.batch_size = batch_size

        self.encoder = CustomFlow(image_shape=(self.image_height, self.image_width, 6), batch_size=batch_size, pretrained=True).build_model()
        self.encoder.build((self.batch_size, self.image_height, self.image_width, 6))
        self.encoder.trainable = True
        
        # 공통 특징 추출층
        self.shared_features_1 = keras.Sequential([
            std_conv(1, 256, 1, use_bias=True, name='shared_conv1'),
            keras.layers.LeakyReLU(),
            std_conv(3, 256, 1, use_bias=True, name='shared_conv1_2'),
            keras.layers.LeakyReLU(),
        ])

        self.shared_features_2 = keras.Sequential([
            std_conv(3, 256, 1, use_bias=True, name='shared_conv2'),
            # keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
        ])

        self.shared_features_3 = keras.Sequential([
            std_conv(3, 6, 1, use_bias=True, name='shared_conv3'),
            # keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
        ]) 

        # 밝기 조정 파라미터 브랜치 (a와 b)
        self.a_conv = keras.layers.Conv2D(
            filters=1, kernel_size=(1, 1), strides=(1, 1),
            padding='same', name='a_conv'
        )
        
        self.b_conv = keras.layers.Conv2D(
            filters=1, kernel_size=(1, 1), strides=(1, 1),
            padding='same', name='b_conv'
        )

        self.global_pool = keras.layers.GlobalAveragePooling2D()

    def call(self, inputs, training=False):
        x = self.encoder(inputs, training=training)
        shared_1 = self.shared_features_1(x)
        shared_2 = self.shared_features_2(shared_1)
        shared_3 = self.shared_features_3(shared_2)

        out_pose = tf.reduce_mean(shared_3, axis=[1, 2], keepdims=False)

        out_a = self.a_conv(shared_2)
        out_a = tf.math.softplus(out_a) # softplus activation
        out_a = tf.reduce_mean(out_a, axis=[1, 2], keepdims=False)

        out_b = self.b_conv(shared_2)
        out_b = tf.math.tanh(out_b) # tanh activation
        out_b = tf.reduce_mean(out_b, axis=[1, 2], keepdims=False)

        out_pose *= 0.01
        out_a *= 0.01
        out_b *= 0.01

        return out_pose, out_a, out_b


class PoseNetExtra(keras.Model):
    """
    - 입력: (B, H, W, 6)  (ex: 소스+타겟 concat)
    - 내부: ResNet-18 인코더 -> Conv/ReduceMean -> Reshape -> scale
    - 출력: (B, 1, 6)  (Monodepth2식 pose)
    """
    def __init__(self,
                 image_shape: tuple,
                 batch_size: int,
                 prefix='pose_resnet',
                 **kwargs):
        super(PoseNetExtra, self).__init__(**kwargs)
        
        self.image_height = image_shape[0]
        self.image_width = image_shape[1]
        self.batch_size = batch_size
        self.pose_scale = 0.1

        self.encoder = CustomFlow(image_shape=(self.image_height, self.image_width, 6), batch_size=batch_size, pretrained=True).build_model()
        self.encoder.build((self.batch_size, self.image_height, self.image_width, 6))
        self.encoder.trainable = True

        # self.encoder = Resnet(image_shape=(self.image_height, self.image_width, 6), batch_size=batch_size, pretrained=True, prefix='resnet18_pose').build_model()
        # self.encoder.build((self.batch_size, self.image_height, self.image_width, 6))
        # self.encoder.trainable = True
        
        # 공통 특징 추출층
        self.shared_features = keras.Sequential([
            std_conv(1, 256, 1, use_bias=False, name='shared_conv1'),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
            std_conv(3, 256, 1, use_bias=False, name='shared_conv2'),
            keras.layers.BatchNormalization(),
            keras.layers.LeakyReLU(),
            std_conv(3, 256, 1, use_bias=True, name='shared_conv3'),
            keras.layers.LeakyReLU(),
        ])

        self.global_pool = keras.layers.GlobalAveragePooling2D()

        # 회전 브랜치
        self.rotation_branch = keras.Sequential([
            keras.layers.Dense(256),
            keras.layers.LeakyReLU(),
            keras.layers.Dense(3)  # 축-각도 또는 오일러 각도
        ])
        
        # 이동 브랜치
        self.translation_branch = keras.Sequential([
            keras.layers.Dense(256),
            keras.layers.LeakyReLU(),
            keras.layers.Dense(3)  # XYZ 이동
        ])

    def call(self, inputs, training=False):
        x = self.encoder(inputs, training=training)
        features = self.shared_features(x)

        features = self.global_pool(features)

        # 분리된 예측
        rotation = self.rotation_branch(features) * 0.01  # 회전에 적합한 스케일링
        translation = self.translation_branch(features) * 0.01  # 이동에 적합한 스케일링
        
        # 결합된 포즈 벡터
        return tf.concat([rotation, translation], axis=-1)


class ImprovedPoseNet(keras.Model):
    def __init__(self,
                 image_shape: tuple,
                 batch_size: int,
                 num_iterations: int = 3,
                 use_correlation: bool = True,
                 prefix='pose_resnet',
                 **kwargs):
        super(ImprovedPoseNet, self).__init__(**kwargs)
        
        self.image_height = image_shape[0]
        self.image_width = image_shape[1]
        self.batch_size = batch_size
        self.num_iterations = num_iterations
        self.use_correlation = use_correlation

        # Encoder for feature extraction
        self.encoder = Resnet(image_shape=(self.image_height, self.image_width, 6), 
                            batch_size=batch_size, pretrained=True, 
                            prefix='resnet18_pose').build_model()
        self.encoder.build((self.batch_size, self.image_height, self.image_width, 6))
        self.encoder.trainable = True
        
        # Initial pose estimation (keep existing structure)
        self.pose_conv0 = std_conv(1, 256, 1, name='pose_conv0')
        self.pose_act0 = keras.layers.ReLU(name='pose_relu0')
        self.pose_conv1 = std_conv(3, 256, 1, name='pose_conv1')
        self.pose_act1 = keras.layers.ReLU(name='pose_relu1')
        self.pose_conv2 = std_conv(3, 256, 1, name='pose_conv2')
        self.pose_act2 = keras.layers.ReLU(name='pose_relu2')
        
        # Initial pose prediction
        self.pose_conv3 = keras.layers.Conv2D(
            filters=6, kernel_size=(1, 1), strides=(1, 1),
            activation=None, name='pose_conv3'
        )
        
        # Define fixed hidden state size
        self.hidden_size = 256
        
        # Feature projection to fixed size
        self.feature_projection = keras.layers.Dense(
            self.hidden_size, 
            activation='relu',
            name='feature_projection'
        )
        
        # GRU input projection layer
        self.gru_input_projection = keras.layers.Dense(
            self.hidden_size,
            activation=None,
            name='gru_input_projection'
        )
        
        # GRU for iterative refinement
        self.gru_cell = keras.layers.GRUCell(
            units=self.hidden_size,
            name='pose_gru'
        )
        
        # Refinement head
        self.refinement_dense1 = keras.layers.Dense(128, activation='relu', name='refine_dense1')
        self.refinement_dense2 = keras.layers.Dense(6, activation=None, name='refine_dense2')
        
        # Correlation layer (optional)
        if self.use_correlation:
            self.correlation_conv = keras.layers.Conv2D(
                filters=64, kernel_size=(1, 1), strides=(1, 1),
                activation='relu', name='correlation_conv'
            )

    def compute_correlation(self, feat1, feat2):
        """Compute correlation between two feature maps"""
        # Normalize features
        feat1_norm = tf.nn.l2_normalize(feat1, axis=-1)
        feat2_norm = tf.nn.l2_normalize(feat2, axis=-1)
        
        # Compute dot product correlation
        correlation = tf.reduce_sum(feat1_norm * feat2_norm, axis=-1, keepdims=True)
        
        return correlation

    def call(self, inputs, training=False):
        # Extract features
        features, _ = self.encoder(inputs, training=training)
        
        # Initial pose estimation
        x = self.pose_conv0(features)
        x = self.pose_act0(x)
        x = self.pose_conv1(x)
        x = self.pose_act1(x)
        x = self.pose_conv2(x)
        x = self.pose_act2(x)
        x = self.pose_conv3(x)
        
        # Global average pooling
        x_pooled = tf.reduce_mean(x, axis=[1, 2])  # [B, 6]
        initial_pose = x_pooled * 0.01  # scale
        
        # Prepare features for GRU
        feat_pooled = tf.reduce_mean(features, axis=[1, 2])  # [B, C]
        
        # Compute correlation if enabled
        if self.use_correlation:
            # Split features for two frames along channel dimension
            c = features.shape[-1]
            feat_t_minus_1 = features[..., :c//2]
            feat_t = features[..., c//2:]
            
            correlation = self.compute_correlation(feat_t_minus_1, feat_t)
            corr_feat = self.correlation_conv(correlation)
            corr_pooled = tf.reduce_mean(corr_feat, axis=[1, 2])
            feat_pooled = tf.concat([feat_pooled, corr_pooled], axis=-1)
        
        # Project features to fixed hidden size
        hidden_state = self.feature_projection(feat_pooled)  # [B, hidden_size]
        
        # GRU-based iterative refinement
        pose = initial_pose
        
        for _ in range(self.num_iterations):
            # Concatenate current pose with hidden state
            gru_input = tf.concat([hidden_state, pose], axis=-1)  # [B, hidden_size + 6]
            
            # Project to match GRU input size
            gru_input_projected = self.gru_input_projection(gru_input)
            
            # GRU update
            hidden_state, _ = self.gru_cell(gru_input_projected, [hidden_state])
            
            # Predict pose update
            delta_pose = self.refinement_dense1(hidden_state)
            delta_pose = self.refinement_dense2(delta_pose)
            
            # Update pose with small step size
            pose = pose + 0.1 * delta_pose
        
        # Estimate uncertainty
        # uncertainty = self.uncertainty_dense(hidden_state)
        
        return pose

class SimplePoseNetWithRefinement(keras.Model):
    def __init__(self,
                 image_shape: tuple,
                 batch_size: int,
                 num_iterations: int = 3,
                 prefix='pose_resnet',
                 **kwargs):
        super(SimplePoseNetWithRefinement, self).__init__(**kwargs)
        
        self.image_height = image_shape[0]
        self.image_width = image_shape[1]
        self.batch_size = batch_size
        self.num_iterations = num_iterations

        # Encoder
        self.encoder = Resnet(
            image_shape=(self.image_height, self.image_width, 6), 
            batch_size=batch_size, 
            pretrained=True, 
            prefix='resnet18_pose'
        ).build_model()
        
        # Initial pose estimation
        self.pose_layers = keras.Sequential([
            std_conv(1, 256, 1, name='pose_conv1'),
            keras.layers.ReLU(),
            std_conv(3, 256, 1, name='pose_conv2'),
            keras.layers.ReLU(),
            std_conv(3, 256, 1, name='pose_conv3'),
            keras.layers.ReLU(),
            keras.layers.Conv2D(6, kernel_size=1, activation=None),
            keras.layers.GlobalAveragePooling2D()
        ], name='initial_pose')
        
        # Refinement network
        self.refinement_net = keras.Sequential([
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.Dense(6, activation=None)
        ], name='refinement')

    def call(self, inputs, training=False):
        # Extract features
        features, _ = self.encoder(inputs, training=training)
        
        # Initial pose
        pose = self.pose_layers(features) * 0.01
        
        # Get global features for refinement
        feat_global = tf.reduce_mean(features, axis=[1, 2])
        
        # Iterative refinement
        for _ in range(self.num_iterations):
            # Concatenate pose and features
            refine_input = tf.concat([feat_global, pose], axis=-1)
            
            # Predict update
            delta = self.refinement_net(refine_input)
            
            # Update pose
            pose = pose + 0.1 * delta
        
        return pose

class PoseNet(keras.Model):
    def __init__(self,
                 image_shape: tuple,
                 batch_size: int,
                 prefix='pose_resnet',
                 **kwargs):
        super(PoseNet, self).__init__(**kwargs)
        
        self.image_height = image_shape[0]
        self.image_width = image_shape[1]
        self.batch_size = batch_size

        self.encoder = Resnet(image_shape=(self.image_height, self.image_width, 6), batch_size=batch_size, pretrained=True, prefix='resnet18_pose').build_model()
        self.encoder.build((self.batch_size, self.image_height, self.image_width, 6))
        self.encoder.trainable = True
        
        # filter_size, out_channel, stride, pad='same', name='conv'
        self.pose_conv0 = std_conv(1, 256, 1, name='pose_conv0')  # kernel=1
        self.pose_act0 = keras.layers.ReLU(name='pose_relu')

        self.pose_conv1 = std_conv(3, 256, 1, name='pose_conv1')  # kernel=3
        self.pose_act1 = keras.layers.ReLU(name='pose_relu1')

        self.pose_conv2 = std_conv(3, 256, 1, name='pose_conv2')  # kernel=3
        self.pose_act2 = keras.layers.ReLU(name='pose_relu2')
    
        self.pose_conv3 = keras.layers.Conv2D(
            filters=6, kernel_size=(1, 1), strides=(1, 1),
            activation=None, name='pose_conv3'
        )

        self.reshape_layer = keras.layers.Reshape((6,), name='pose_reshape')

    def call(self, inputs, training=False):
        x, _ = self.encoder(inputs, training=training) 

        x = self.pose_conv0(x)
        x = self.pose_act0(x)

        x = self.pose_conv1(x)
        x = self.pose_act1(x)

        x = self.pose_conv2(x)
        x = self.pose_act2(x)
            
        x = self.pose_conv3(x)

        x = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        x = self.reshape_layer(x)# [B, 6]
        x = x * 0.01 # scale
        return x
    
if __name__ == '__main__':
    # Test PoseNet
    image_shape = (480, 640)
    batch_size = 2
    # posenet = PoseNetExtra(image_shape=image_shape, batch_size=batch_size)
    posenet = PoseNetAB(image_shape=image_shape, batch_size=batch_size)
    posenet.build((batch_size, image_shape[0], image_shape[1], 6))
    posenet.summary()
    
    # Test forward
    inputs = tf.random.normal((batch_size, image_shape[0], image_shape[1], 6))
    outputs = posenet(inputs)
    print(outputs)