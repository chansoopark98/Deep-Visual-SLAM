import tensorflow as tf
import math

class MultiQueryAttentionLayerV2(tf.keras.layers.Layer):
    """Multi-Query Self-Attention layer with optional spatial downsampling."""
    def __init__(self, in_channels, out_channels, num_heads, key_dim, value_dim,
                 query_h_strides=1, query_w_strides=1, kv_strides=1,
                 downsampling_kernel_size=3, dropout=0.0, use_bias=False,
                 norm_momentum=0.99, norm_epsilon=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.key_dim = key_dim
        self.value_dim = value_dim
        self.query_h_strides = query_h_strides
        self.query_w_strides = query_w_strides
        self.kv_strides = kv_strides
        # Layers for query projection (with optional downsampling via pooling)
        self.query_pool = None
        if query_h_strides > 1 or query_w_strides > 1:
            # Average pooling to reduce query resolution (if specified)
            self.query_pool = tf.keras.layers.AveragePooling2D(pool_size=(query_h_strides, query_w_strides))
        # We apply normalization on the input before query projection (pre-norm)
        self.query_norm = tf.keras.layers.BatchNormalization(momentum=norm_momentum, epsilon=norm_epsilon)
        # 1x1 conv to project input to queries of shape [num_heads * key_dim]
        self.query_proj = tf.keras.layers.Conv2D(filters=num_heads * key_dim, kernel_size=1,
                                                 strides=1, use_bias=use_bias)
        # Layers for key projection
        self.key_dw = None
        if kv_strides > 1:
            # Depthwise conv for downsampling keys (stride=kv_strides, kernel=downsampling_kernel_size)
            self.key_dw = tf.keras.layers.DepthwiseConv2D(kernel_size=downsampling_kernel_size,
                                                          strides=kv_strides, padding='same', use_bias=False)
        # Normalize after depthwise conv (or identity) for keys
        self.key_norm = tf.keras.layers.BatchNormalization(momentum=norm_momentum, epsilon=norm_epsilon)
        # 1x1 conv to project to shared key (output channels = num_heads * key_dim to match queries concatenated)
        self.key_proj = tf.keras.layers.Conv2D(filters=num_heads * key_dim, kernel_size=1,
                                               strides=1, use_bias=use_bias)
        # Layers for value projection
        self.value_dw = None
        if kv_strides > 1:
            # Depthwise conv for downsampling values
            self.value_dw = tf.keras.layers.DepthwiseConv2D(kernel_size=downsampling_kernel_size,
                                                            strides=kv_strides, padding='same', use_bias=False)
        self.value_norm = tf.keras.layers.BatchNormalization(momentum=norm_momentum, epsilon=norm_epsilon)
        self.value_proj = tf.keras.layers.Conv2D(filters=num_heads * value_dim, kernel_size=1,
                                                 strides=1, use_bias=use_bias)
        # Output projection layers
        self.upsample = None
        if query_h_strides > 1 or query_w_strides > 1:
            # If queries were pooled, upsample outputs back to original spatial size
            self.upsample = tf.keras.layers.UpSampling2D(size=(query_h_strides, query_w_strides),
                                                         interpolation='bilinear')
        # 1x1 conv to project concatenated heads to desired output channels
        self.output_proj = tf.keras.layers.Conv2D(filters=out_channels, kernel_size=1,
                                                  strides=1, use_bias=use_bias)
        # Dropout for attention weights (if any)
        self.attn_dropout = tf.keras.layers.Dropout(dropout) if dropout > 0.0 else None

    def call(self, x, training=False):
        # Input shape: [B, H, W, C_in]
        # Optional query downsampling
        q_in = x
        if self.query_pool:
            q_in = self.query_pool(q_in)
        # Normalize then project queries
        q_norm = self.query_norm(q_in, training=training)
        q = self.query_proj(q_norm)  # shape: [B, Hq, Wq, num_heads * key_dim]
        # Project keys: use downsample conv if specified, then BN, then conv1x1
        k_in = q  # note: we use the query-projected features as input for key/value projections&#8203;:contentReference[oaicite:23]{index=23}&#8203;:contentReference[oaicite:24]{index=24}
        if self.key_dw:
            k_in = self.key_dw(k_in)  # depthwise conv downsampling for keys&#8203;:contentReference[oaicite:25]{index=25}
        k_norm = self.key_norm(k_in, training=training)
        k = self.key_proj(k_norm)   # shape: [B, Hk, Wk, num_heads * key_dim]
        # Project values similarly
        v_in = q
        if self.value_dw:
            v_in = self.value_dw(v_in)  # depthwise conv downsampling for values&#8203;:contentReference[oaicite:26]{index=26}
        v_norm = self.value_norm(v_in, training=training)
        v = self.value_proj(v_norm)  # shape: [B, Hk, Wk, num_heads * value_dim]
        # Reshape and compute attention:
        # Reshape query, key, value to [B, num_heads, seq_len, dim]
        B = tf.shape(q)[0]
        Hq = tf.shape(q)[1]; Wq = tf.shape(q)[2]
        Hk = tf.shape(k)[1]; Wk = tf.shape(k)[2]
        # Flatten spatial dims
        # q_flat: [B, Hq*Wq, num_heads, key_dim] -> transpose to [B, num_heads, Hq*Wq, key_dim]
        q_flat = tf.reshape(q, [B, Hq*Wq, self.num_heads, self.key_dim])
        q_flat = tf.transpose(q_flat, [0, 2, 1, 3])
        # k_flat: [B, num_heads, Hk*Wk, key_dim]
        k_flat = tf.reshape(k, [B, Hk*Wk, self.num_heads, self.key_dim])
        k_flat = tf.transpose(k_flat, [0, 2, 1, 3])
        # v_flat: [B, num_heads, Hk*Wk, value_dim]
        v_flat = tf.reshape(v, [B, Hk*Wk, self.num_heads, self.value_dim])
        v_flat = tf.transpose(v_flat, [0, 2, 1, 3])
        # Scaled dot-product attention: scores [B, heads, Q_len, K_len]
        scores = tf.matmul(q_flat, k_flat, transpose_b=True) / tf.math.sqrt(tf.cast(self.key_dim, tf.float32))
        attn_weights = tf.nn.softmax(scores, axis=-1)
        if self.attn_dropout:
            attn_weights = self.attn_dropout(attn_weights, training=training)  # dropout on attention map&#8203;:contentReference[oaicite:27]{index=27}
        # Attention output: [B, heads, Q_len, value_dim]
        context = tf.matmul(attn_weights, v_flat)
        # Reshape back to image spatial shape
        context = tf.transpose(context, [0, 2, 1, 3])  # -> [B, Q_len, heads, value_dim]
        context = tf.reshape(context, [B, Hq, Wq, self.num_heads * self.value_dim])
        # Upsample if queries were pooled
        if self.upsample:
            context = self.upsample(context)
        # Project to output channels
        output = self.output_proj(context)  # shape: [B, H_out, W_out, out_channels]
        return output

# Alias for clarity (same implementation in this context)
OptimizedMultiQueryAttentionLayerWithDownSampling = MultiQueryAttentionLayerV2

class MultiHeadSelfAttentionBlock(tf.keras.layers.Layer):
    """Block that applies multi-head self-attention (MQA or standard) with normalization, residual, and layer scale."""
    def __init__(self, input_dim, output_dim, num_heads, key_dim, value_dim,
                 use_multi_query=True, query_h_strides=1, query_w_strides=1, kv_strides=1,
                 downsampling_dw_kernel_size=3, dropout=0.0, use_bias=False,
                 use_layer_scale=True, layer_scale_init_value=1e-5,
                 use_residual=True, norm_momentum=0.99, norm_epsilon=1e-3, **kwargs):
        super().__init__(**kwargs)
        self.use_multi_query = use_multi_query
        self.use_residual = use_residual
        # Normalization (BatchNorm) before attention (pre-norm)
        self.norm = tf.keras.layers.BatchNormalization(momentum=norm_momentum, epsilon=norm_epsilon)
        if use_multi_query:
            # Use optimized MQA block
            self.attention = OptimizedMultiQueryAttentionLayerWithDownSampling(
                in_channels=input_dim, out_channels=output_dim, num_heads=num_heads,
                key_dim=key_dim, value_dim=value_dim,
                query_h_strides=query_h_strides, query_w_strides=query_w_strides, kv_strides=kv_strides,
                downsampling_kernel_size=downsampling_dw_kernel_size, dropout=dropout,
                use_bias=use_bias, norm_momentum=norm_momentum, norm_epsilon=norm_epsilon)
        else:
            # Standard multi-head self-attention (queries=keys=values)
            # We will use tf.keras.layers.MultiHeadAttention on flattened spatial dimensions
            self.mha = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, value_dim=value_dim,
                                                          dropout=dropout, output_shape=output_dim)
        # Layer scale parameter (if enabled)
        self.use_layer_scale = use_layer_scale
        if use_layer_scale:
            # Trainable scalar for scaling the block's output (initialized to a small value)
            self.gamma = self.add_weight(name="layer_scale", shape=(output_dim,),
                                         initializer=tf.constant_initializer(layer_scale_init_value),
                                         trainable=True)
        else:
            self.gamma = None

    def call(self, inputs, training=False):
        # Save shortcut for residual
        shortcut = inputs
        # Pre-normalization
        x = self.norm(inputs, training=training)
        # Apply attention
        if self.use_multi_query:
            x = self.attention(x, training=training)  # MQA block output&#8203;:contentReference[oaicite:28]{index=28}&#8203;:contentReference[oaicite:29]{index=29}
        else:
            # Flatten spatial dimensions for MultiHeadAttention
            B, H, W, C = tf.shape(x)[0], tf.shape(x)[1], tf.shape(x)[2], tf.shape(x)[3]
            seq_len = H * W
            x_flat = tf.reshape(x, [B, seq_len, C])
            # Self-attention (uses queries = keys = values = x_flat)
            x_flat = self.mha(x_flat, x_flat, x_flat, training=training)  # shape [B, seq_len, output_dim]
            # Restore spatial dimensions
            x = tf.reshape(x_flat, [B, H, W, -1])
        # Apply layer scale if enabled
        if self.use_layer_scale:
            x = self.gamma * x  # scale the attention output&#8203;:contentReference[oaicite:30]{index=30}
        # Residual connection
        if self.use_residual:
            x = x + shortcut  # add input (shortcut) to output&#8203;:contentReference[oaicite:31]{index=31}
        return x
        
# Utility: make number of channels divisible by 8 (as in official MobileNetV4)&#8203;:contentReference[oaicite:9]{index=9}
def make_divisible(v, divisor=8):
    min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    # prevent rounding down by more than 10%
    if new_v < 0.9 * v:
        new_v += divisor
    return int(new_v)

# Activation: ReLU6 (to match MobileNetV4 use of ReLU6 in conv layers)&#8203;:contentReference[oaicite:10]{index=10}
relu6 = tf.keras.layers.ReLU(max_value=6.0)

# Convolution block: Conv2D with BatchNorm (if norm=True) and ReLU6 (if act=True)
def conv_bn_act(x, filters, kernel_size=1, strides=1, groups=1, norm=True, act=True):
    # Conv2D (depthwise if groups==filters==input_channels)
    if groups == filters == x.shape[-1]:
        # Depthwise conv
        x = tf.keras.layers.DepthwiseConv2D(kernel_size, strides=strides, padding='same', 
                                            depth_multiplier=1, use_bias=not norm)(x)
    else:
        x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='same', 
                                   groups=groups, use_bias=not norm)(x)
    if norm:
        x = tf.keras.layers.BatchNormalization()(x)
    if act:
        x = relu6(x)
    return x

# Fused Inverted Residual Block (MobileNetV3-style fused MBConv, without SE)&#8203;:contentReference[oaicite:11]{index=11}&#8203;:contentReference[oaicite:12]{index=12}
def fused_inverted_block(x, inp_channels, out_channels, stride, expand_ratio, act_final=False):
    shortcut = x
    hidden_dim = int(round(inp_channels * expand_ratio))
    # Expansion (fused conv 3x3)
    if expand_ratio != 1:
        # Use groups=1 (normal conv) with kernel 3
        x = conv_bn_act(x, hidden_dim, kernel_size=3, strides=stride, groups=1, norm=True, act=True)
    else:
        # If no expansion, we still need to handle stride by depthwise if stride>1
        # (Not explicitly in fused block definition, expansion=1 case rarely used)
        if stride == 2:
            # Depthwise conv just for downsampling if needed
            x = conv_bn_act(x, hidden_dim, kernel_size=3, strides=stride, groups=hidden_dim, norm=True, act=True)
    # No SE (squeeze-and-excite) in MobileNetV4 conv variants
    # Projection 1x1
    x = conv_bn_act(x, out_channels, kernel_size=1, strides=1, groups=1, norm=True, act=act_final)
    # Residual connection if shapes match
    if stride == 1 and inp_channels == out_channels:
        x = tf.keras.layers.Add()([shortcut, x])
    return x

# Universal Inverted Bottleneck Block (UIB)&#8203;:contentReference[oaicite:13]{index=13}&#8203;:contentReference[oaicite:14]{index=14}
def universal_inverted_block(x, inp_channels, out_channels, start_dw_kernel, middle_dw_kernel, middle_dw_downsample, stride, expand_ratio):
    # Optional starting depthwise conv
    if start_dw_kernel and start_dw_kernel > 0:
        # If middle_dw_downsample is True, do not downsample at start (do at middle instead)&#8203;:contentReference[oaicite:15]{index=15}
        stride_start = stride if not middle_dw_downsample else 1
        # Depthwise conv on input
        x = conv_bn_act(x, inp_channels, kernel_size=start_dw_kernel, strides=stride_start, 
                        groups=inp_channels, norm=True, act=False)  # no activation on start dw&#8203;:contentReference[oaicite:16]{index=16}
    # 1x1 expansion conv
    hidden_dim = make_divisible(inp_channels * expand_ratio, 8)
    x = conv_bn_act(x, hidden_dim, kernel_size=1, strides=1, groups=1, norm=True, act=True)
    # Optional middle depthwise conv
    if middle_dw_kernel and middle_dw_kernel > 0:
        # If middle_dw_downsample is True, use stride for this depthwise (otherwise it was used at start)&#8203;:contentReference[oaicite:17]{index=17}
        stride_mid = stride if middle_dw_downsample else 1
        x = conv_bn_act(x, hidden_dim, kernel_size=middle_dw_kernel, strides=stride_mid, 
                        groups=hidden_dim, norm=True, act=True)
    # 1x1 projection conv (no activation)&#8203;:contentReference[oaicite:18]{index=18}
    x = conv_bn_act(x, out_channels, kernel_size=1, strides=1, groups=1, norm=True, act=False)
    return x  # no residual added inside UIB (residuals handled via attention blocks if any)

# Mobile Multi-Query Self-Attention Block (Mobile MQA)&#8203;:contentReference[oaicite:19]{index=19}&#8203;:contentReference[oaicite:20]{index=20}
def attention_block(x, num_heads, key_dim, value_dim, kv_strides=1, use_layer_scale=True, use_residual=True):
    # Save shortcut for residual
    shortcut = x
    # Layer normalization (BatchNorm2D used in the original for input norm)&#8203;:contentReference[oaicite:21]{index=21}
    x_norm = tf.keras.layers.BatchNormalization()(x)
    # If downsampling keys/values spatially
    if kv_strides > 1:
        # Downsample keys and values by average pooling
        k_v = tf.keras.layers.AveragePooling2D(pool_size=(kv_strides, kv_strides), strides=kv_strides)(x_norm)
    else:
        k_v = x_norm
    q = x_norm  # query uses full resolution (query strides = 1 in this implementation)
    # Use Keras MultiHeadAttention to perform attention
    # We flatten spatial dimensions to sequence length for attention
    q_shape = tf.shape(q)  # dynamic shape
    k_shape = tf.shape(k_v)
    # Reshape [B, H, W, C] -> [B, seq_len, C] for MHA
    q_flat = tf.keras.layers.Reshape((-1, int(q.shape[-1])))(q)
    k_flat = tf.keras.layers.Reshape((-1, int(k_v.shape[-1])))(k_v)
    v_flat = tf.keras.layers.Reshape((-1, int(k_v.shape[-1])))(k_v)
    # Multi-head attention (outputs sequence of length = q_seq_len, features = C by default) 
    attn_layer = tf.keras.layers.MultiHeadAttention(num_heads=num_heads, key_dim=key_dim, value_dim=value_dim, output_shape=int(q.shape[-1]))
    attn_out_seq = attn_layer(query=q_flat, key=k_flat, value=v_flat)
    # Reshape back to spatial [B, H, W, C]
    # (We infer H, W from original query tensor)
    # Use tf.reshape with dynamic shape components:
    attn_out = tf.reshape(attn_out_seq, [-1, tf.shape(q)[1], tf.shape(q)[2], int(q.shape[-1])])
    # Optional layer scale (learnable scaling of attention output)&#8203;:contentReference[oaicite:22]{index=22}&#8203;:contentReference[oaicite:23]{index=23}
    if use_layer_scale:
        # Per-channel trainable scale initialized at a small value (1e-5)
        scale = tf.keras.layers.LayerNormalization(axis=-1, epsilon=1e-6, center=False, scale=True)
        # LayerNormalization with center=False, scale=True can be repurposed to simulate layer scale by initializing scale gamma to 1e-5
        # We'll manually set gamma initial value to 1e-5 after building the layer
        # (Since LayerNormalization in TF doesn't allow direct init of gamma, we'll set it via weights if needed.)
        # Alternatively, implement a custom layer:
        gamma = tf.Variable(1e-5 * tf.ones(shape=(attn_out.shape[-1],)), trainable=True)
        attn_out = attn_out * gamma
    # Residual connection
    if use_residual:
        out = tf.keras.layers.Add()([shortcut, attn_out])
    else:
        out = attn_out
    return out

# Define MobileNetV4 ConvSmall
def MobileNetV4ConvSmall(image_shape=(224, 224, 3), num_classes=1000):
    inputs = tf.keras.Input(shape=image_shape)
    # Initial conv layer&#8203;:contentReference[oaicite:24]{index=24}
    x = conv_bn_act(inputs, filters=32, kernel_size=3, strides=2)        # conv0: 3x3 conv, 32 out, stride 2
    stem = x

    # Stage 1&#8203;:contentReference[oaicite:25]{index=25}: 2 ConvBN blocks
    x = conv_bn_act(x, filters=32, kernel_size=3, strides=2)            # layer1 block 1: 3x3 conv, 32 out, stride 2
    x = conv_bn_act(x, filters=32, kernel_size=1, strides=1)            # layer1 block 2: 1x1 conv, 32 out, stride 1
    stage_1 = x

    # Stage 2&#8203;:contentReference[oaicite:26]{index=26}: 2 ConvBN blocks
    x = conv_bn_act(x, filters=96, kernel_size=3, strides=2)            # layer2 block 1: 3x3 conv, 96 out, stride 2
    x = conv_bn_act(x, filters=64, kernel_size=1, strides=1)            # layer2 block 2: 1x1 conv, 64 out, stride 1
    stage_2 = x

    # Stage 3&#8203;:contentReference[oaicite:27]{index=27}: 6 UIB blocks
    # Block 1 (ExtraDW variant): start_dw=5, middle_dw=5, downsample in middle, expand_ratio=3.0&#8203;:contentReference[oaicite:28]{index=28}
    x = universal_inverted_block(x, inp_channels=64, out_channels=96, start_dw_kernel=5, middle_dw_kernel=5, 
                                 middle_dw_downsample=True, stride=2, expand_ratio=3.0)
    # Blocks 2-5 (IB variants): no start_dw, middle_dw=3, expand_ratio=2.0, stride=1&#8203;:contentReference[oaicite:29]{index=29}
    for _ in range(4):
        x = universal_inverted_block(x, inp_channels=96, out_channels=96, start_dw_kernel=0, middle_dw_kernel=3,
                                     middle_dw_downsample=True, stride=1, expand_ratio=2.0)
    # Block 6 (ConvNext variant): start_dw=3, no middle_dw, expand_ratio=4.0, stride=1&#8203;:contentReference[oaicite:30]{index=30}
    x = universal_inverted_block(x, inp_channels=96, out_channels=96, start_dw_kernel=3, middle_dw_kernel=0,
                                 middle_dw_downsample=True, stride=1, expand_ratio=4.0)
    stage_3 = x
    
    # Stage 4&#8203;:contentReference[oaicite:31]{index=31}: 6 UIB blocks
    # Block 1 (ExtraDW variant): start_dw=3, middle_dw=3, expand_ratio=6.0, stride=2&#8203;:contentReference[oaicite:32]{index=32}
    x = universal_inverted_block(x, inp_channels=96, out_channels=128, start_dw_kernel=3, middle_dw_kernel=3,
                                 middle_dw_downsample=True, stride=2, expand_ratio=6.0)
    # Block 2 (ExtraDW): start_dw=5, middle_dw=5, expand_ratio=4.0, stride=1&#8203;:contentReference[oaicite:33]{index=33}
    x = universal_inverted_block(x, inp_channels=128, out_channels=128, start_dw_kernel=5, middle_dw_kernel=5,
                                 middle_dw_downsample=True, stride=1, expand_ratio=4.0)
    # Block 3 (IB): no start_dw, middle_dw=5, expand_ratio=4.0, stride=1
    x = universal_inverted_block(x, inp_channels=128, out_channels=128, start_dw_kernel=0, middle_dw_kernel=5,
                                 middle_dw_downsample=True, stride=1, expand_ratio=4.0)
    # Block 4 (IB): no start_dw, middle_dw=5, expand_ratio=3.0, stride=1
    x = universal_inverted_block(x, inp_channels=128, out_channels=128, start_dw_kernel=0, middle_dw_kernel=5,
                                 middle_dw_downsample=True, stride=1, expand_ratio=3.0)
    # Block 5 (IB): no start_dw, middle_dw=3, expand_ratio=4.0, stride=1
    x = universal_inverted_block(x, inp_channels=128, out_channels=128, start_dw_kernel=0, middle_dw_kernel=3,
                                 middle_dw_downsample=True, stride=1, expand_ratio=4.0)
    # Block 6 (IB): no start_dw, middle_dw=3, expand_ratio=4.0, stride=1&#8203;:contentReference[oaicite:34]{index=34}
    x = universal_inverted_block(x, inp_channels=128, out_channels=128, start_dw_kernel=0, middle_dw_kernel=3,
                                 middle_dw_downsample=True, stride=1, expand_ratio=4.0)
    
    stage_4 = x

    # # Stage 5: Final 1x1 convolutions to expand to 1280&#8203;:contentReference[oaicite:35]{index=35}&#8203;:contentReference[oaicite:36]{index=36}
    # x = conv_bn_act(x, filters=960, kernel_size=1, strides=1, norm=True, act=True)   # 1x1 conv to 960
    # x = tf.keras.layers.GlobalAveragePooling2D()(x)                                 # global average pool
    # x = tf.keras.layers.Reshape((1, 1, 960))(x)  # reshape to 1x1 feature map for final conv
    # x = conv_bn_act(x, filters=1280, kernel_size=1, strides=1, norm=True, act=True)  # 1x1 conv to 1280
    # x = tf.keras.layers.Flatten()(x)
    # outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    skips = [stage_3, stage_2, stage_1, stem]
    return tf.keras.Model(inputs, [stage_4, skips], name="MobileNetV4ConvSmall")

# Define MobileNetV4 ConvMedium
def MobileNetV4ConvMedium(input_shape=(224, 224, 3), num_classes=1000):
    inputs = tf.keras.Input(shape=input_shape)
    x = conv_bn_act(inputs, filters=32, kernel_size=3, strides=2)       # conv0: 32 filters
    # Stage 1&#8203;:contentReference[oaicite:37]{index=37}: fused inverted block (single block)
    x = fused_inverted_block(x, inp_channels=32, out_channels=48, stride=2, expand_ratio=4.0, act_final=True)  # 3x3 fused, stride 2
    # Stage 2&#8203;:contentReference[oaicite:38]{index=38}: UIB blocks (2 blocks)
    x = universal_inverted_block(x, inp_channels=48, out_channels=80, start_dw_kernel=3, middle_dw_kernel=5,
                                 middle_dw_downsample=True, stride=2, expand_ratio=4.0)   # downsample block
    x = universal_inverted_block(x, inp_channels=80, out_channels=80, start_dw_kernel=3, middle_dw_kernel=3,
                                 middle_dw_downsample=True, stride=1, expand_ratio=2.0)   # second block, output stage2
    # Stage 3&#8203;:contentReference[oaicite:39]{index=39}: UIB blocks (8 blocks)
    # Block 1: ExtraDW variant (start_dw=3, middle_dw=5, expand_ratio=6.0, stride=2)
    x = universal_inverted_block(x, inp_channels=80, out_channels=160, start_dw_kernel=3, middle_dw_kernel=5,
                                 middle_dw_downsample=True, stride=2, expand_ratio=6.0)
    # Blocks 2-3: IB variants (start_dw=3, middle_dw=3, expand_ratio=4.0, stride=1) – do twice
    for _ in range(2):
        x = universal_inverted_block(x, inp_channels=160, out_channels=160, start_dw_kernel=3, middle_dw_kernel=3,
                                     middle_dw_downsample=True, stride=1, expand_ratio=4.0)
    # Block 4: ExtraDW variant (start_dw=3, middle_dw=5, expand_ratio=4.0, stride=1)
    x = universal_inverted_block(x, inp_channels=160, out_channels=160, start_dw_kernel=3, middle_dw_kernel=5,
                                 middle_dw_downsample=True, stride=1, expand_ratio=4.0)
    # Block 5: IB variant (start_dw=3, middle_dw=3, expand_ratio=4.0, stride=1)
    x = universal_inverted_block(x, inp_channels=160, out_channels=160, start_dw_kernel=3, middle_dw_kernel=3,
                                 middle_dw_downsample=True, stride=1, expand_ratio=4.0)
    # Block 6: ConvNext variant (start_dw=3, middle_dw=0, expand_ratio=4.0, stride=1)
    x = universal_inverted_block(x, inp_channels=160, out_channels=160, start_dw_kernel=3, middle_dw_kernel=0,
                                 middle_dw_downsample=True, stride=1, expand_ratio=4.0)
    # Block 7: FFN variant (start_dw=0, middle_dw=0, expand_ratio=2.0, stride=1) – no depthwise
    x = universal_inverted_block(x, inp_channels=160, out_channels=160, start_dw_kernel=0, middle_dw_kernel=0,
                                 middle_dw_downsample=True, stride=1, expand_ratio=2.0)
    # Block 8: ConvNext variant (start_dw=3, middle_dw=0, expand_ratio=4.0, stride=1)&#8203;:contentReference[oaicite:40]{index=40}
    x = universal_inverted_block(x, inp_channels=160, out_channels=160, start_dw_kernel=3, middle_dw_kernel=0,
                                 middle_dw_downsample=True, stride=1, expand_ratio=4.0)
    # Stage 4&#8203;:contentReference[oaicite:41]{index=41}&#8203;:contentReference[oaicite:42]{index=42}: UIB blocks (11 blocks)
    # Block 1: ExtraDW variant (start_dw=5, middle_dw=5, expand_ratio=6.0, stride=2)
    x = universal_inverted_block(x, inp_channels=160, out_channels=256, start_dw_kernel=5, middle_dw_kernel=5,
                                 middle_dw_downsample=True, stride=2, expand_ratio=6.0)
    # Block 2: ExtraDW (start_dw=5, middle_dw=5, expand_ratio=4.0, stride=1)
    x = universal_inverted_block(x, inp_channels=256, out_channels=256, start_dw_kernel=5, middle_dw_kernel=5,
                                 middle_dw_downsample=True, stride=1, expand_ratio=4.0)
    # Block 3-4: IB variants (start_dw=3, middle_dw=5, expand_ratio=4.0, stride=1) – do twice
    for _ in range(2):
        x = universal_inverted_block(x, inp_channels=256, out_channels=256, start_dw_kernel=3, middle_dw_kernel=5,
                                     middle_dw_downsample=True, stride=1, expand_ratio=4.0)
    # Block 5-6: FFN variants (start_dw=0, middle_dw=0, expand_ratio=4.0, stride=1) – do twice
    for _ in range(2):
        x = universal_inverted_block(x, inp_channels=256, out_channels=256, start_dw_kernel=0, middle_dw_kernel=0,
                                     middle_dw_downsample=True, stride=1, expand_ratio=4.0)
    # Block 7: IB variant (start_dw=3, middle_dw=5, expand_ratio=2.0, stride=1)
    x = universal_inverted_block(x, inp_channels=256, out_channels=256, start_dw_kernel=3, middle_dw_kernel=5,
                                 middle_dw_downsample=True, stride=1, expand_ratio=2.0)
    # Block 8: ExtraDW (start_dw=5, middle_dw=5, expand_ratio=4.0, stride=1)
    x = universal_inverted_block(x, inp_channels=256, out_channels=256, start_dw_kernel=5, middle_dw_kernel=5,
                                 middle_dw_downsample=True, stride=1, expand_ratio=4.0)
    # Block 9-10: FFN variants (start_dw=0, middle_dw=0, expand_ratio=2.0, stride=1) – do twice
    for _ in range(2):
        x = universal_inverted_block(x, inp_channels=256, out_channels=256, start_dw_kernel=0, middle_dw_kernel=0,
                                     middle_dw_downsample=True, stride=1, expand_ratio=2.0)
    # Block 11: ConvNext variant (start_dw=5, middle_dw=0, expand_ratio=2.0, stride=1)&#8203;:contentReference[oaicite:43]{index=43}
    x = universal_inverted_block(x, inp_channels=256, out_channels=256, start_dw_kernel=5, middle_dw_kernel=0,
                                 middle_dw_downsample=True, stride=1, expand_ratio=2.0)
    # Final stage: 1x1 convs to 1280
    x = conv_bn_act(x, filters=960, kernel_size=1, strides=1, norm=True, act=True)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Reshape((1, 1, 960))(x)
    x = conv_bn_act(x, filters=1280, kernel_size=1, strides=1, norm=True, act=True)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs, name="MobileNetV4ConvMedium")

# Define MobileNetV4 ConvLarge
def MobileNetV4ConvLarge(input_shape=(224, 224, 3), num_classes=1000):
    inputs = tf.keras.Input(shape=input_shape)
    x = conv_bn_act(inputs, filters=24, kernel_size=3, strides=2)      # conv0: 24 filters
    # Stage 1&#8203;:contentReference[oaicite:44]{index=44}: fused inverted block
    x = fused_inverted_block(x, inp_channels=24, out_channels=48, stride=2, expand_ratio=4.0, act_final=True)
    # Stage 2&#8203;:contentReference[oaicite:45]{index=45}: UIB blocks (2 blocks)
    x = universal_inverted_block(x, inp_channels=48, out_channels=96, start_dw_kernel=3, middle_dw_kernel=5,
                                 middle_dw_downsample=True, stride=2, expand_ratio=4.0)
    x = universal_inverted_block(x, inp_channels=96, out_channels=96, start_dw_kernel=3, middle_dw_kernel=3,
                                 middle_dw_downsample=True, stride=1, expand_ratio=4.0)
    # Stage 3&#8203;:contentReference[oaicite:46]{index=46}&#8203;:contentReference[oaicite:47]{index=47}: UIB blocks (11 blocks)
    # Block 1: ExtraDW (start_dw=3, middle_dw=5, expand_ratio=4.0, stride=2)
    x = universal_inverted_block(x, inp_channels=96, out_channels=192, start_dw_kernel=3, middle_dw_kernel=5,
                                 middle_dw_downsample=True, stride=2, expand_ratio=4.0)
    # Blocks 2-5: IB (start_dw=3, middle_dw=3, expand_ratio=4.0, stride=1) – 4 blocks
    for _ in range(4):
        x = universal_inverted_block(x, inp_channels=192, out_channels=192, start_dw_kernel=3, middle_dw_kernel=3,
                                     middle_dw_downsample=True, stride=1, expand_ratio=4.0)
    # Blocks 6-10: ExtraDW (start_dw=5, middle_dw=3, expand_ratio=4.0, stride=1) – 5 blocks
    for _ in range(5):
        x = universal_inverted_block(x, inp_channels=192, out_channels=192, start_dw_kernel=5, middle_dw_kernel=3,
                                     middle_dw_downsample=True, stride=1, expand_ratio=4.0)
    # Block 11: ConvNext (start_dw=3, middle_dw=0, expand_ratio=4.0, stride=1)&#8203;:contentReference[oaicite:48]{index=48}
    x = universal_inverted_block(x, inp_channels=192, out_channels=192, start_dw_kernel=3, middle_dw_kernel=0,
                                 middle_dw_downsample=True, stride=1, expand_ratio=4.0)
    # Stage 4&#8203;:contentReference[oaicite:49]{index=49}&#8203;:contentReference[oaicite:50]{index=50}: UIB blocks (13 blocks)
    # Block 1: ExtraDW (start_dw=5, middle_dw=5, expand_ratio=4.0, stride=2)
    x = universal_inverted_block(x, inp_channels=192, out_channels=512, start_dw_kernel=5, middle_dw_kernel=5,
                                 middle_dw_downsample=True, stride=2, expand_ratio=4.0)
    # Blocks 2-5: ExtraDW (start_dw=5, middle_dw=5, expand_ratio=4.0, stride=1) – 4 blocks
    for _ in range(4):
        x = universal_inverted_block(x, inp_channels=512, out_channels=512, start_dw_kernel=5, middle_dw_kernel=5,
                                     middle_dw_downsample=True, stride=1, expand_ratio=4.0)
    # Blocks 6-13: IB/FFN mix (according to spec) – we follow the sequence:
    #  - Blocks 6-7: IB (start_dw=5, middle_dw=0, expand_ratio=4.0)
    for _ in range(2):
        x = universal_inverted_block(x, inp_channels=512, out_channels=512, start_dw_kernel=5, middle_dw_kernel=0,
                                     middle_dw_downsample=True, stride=1, expand_ratio=4.0)
    #  - Blocks 8-9: IB (start_dw=5, middle_dw=3, expand_ratio=4.0)
    for _ in range(2):
        x = universal_inverted_block(x, inp_channels=512, out_channels=512, start_dw_kernel=5, middle_dw_kernel=3,
                                     middle_dw_downsample=True, stride=1, expand_ratio=4.0)
    #  - Blocks 10-12: IB (start_dw=5, middle_dw=0, expand_ratio=4.0)
    for _ in range(3):
        x = universal_inverted_block(x, inp_channels=512, out_channels=512, start_dw_kernel=5, middle_dw_kernel=0,
                                     middle_dw_downsample=True, stride=1, expand_ratio=4.0)
    # Block 13: IB (start_dw=5, middle_dw=0, expand_ratio=4.0)&#8203;:contentReference[oaicite:51]{index=51}
    x = universal_inverted_block(x, inp_channels=512, out_channels=512, start_dw_kernel=5, middle_dw_kernel=0,
                                 middle_dw_downsample=True, stride=1, expand_ratio=4.0)
    # Final stage: 1x1 convs to 1280
    x = conv_bn_act(x, filters=960, kernel_size=1, strides=1, norm=True, act=True)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Reshape((1, 1, 960))(x)
    x = conv_bn_act(x, filters=1280, kernel_size=1, strides=1, norm=True, act=True)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs, name="MobileNetV4ConvLarge")

# Define MobileNetV4 HybridMedium (incorporating attention blocks in stages 4 and 5)
def MobileNetV4HybridMedium(input_shape=(224, 224, 3), num_classes=1000):
    inputs = tf.keras.Input(shape=input_shape)
    x = conv_bn_act(inputs, filters=32, kernel_size=3, strides=2)      # conv0: 32 filters
    x = fused_inverted_block(x, inp_channels=32, out_channels=48, stride=2, expand_ratio=4.0, act_final=True)  # stage1 fused
    # Stage 2: same as ConvMedium stage2 (2 UIB blocks)
    x = universal_inverted_block(x, inp_channels=48, out_channels=80, start_dw_kernel=3, middle_dw_kernel=5,
                                 middle_dw_downsample=True, stride=2, expand_ratio=4.0)
    x = universal_inverted_block(x, inp_channels=80, out_channels=80, start_dw_kernel=3, middle_dw_kernel=3,
                                 middle_dw_downsample=True, stride=1, expand_ratio=2.0)
    # Stage 3: same as ConvMedium stage3 (2 UIB blocks)
    x = universal_inverted_block(x, inp_channels=80, out_channels=160, start_dw_kernel=3, middle_dw_kernel=5,
                                 middle_dw_downsample=True, stride=2, expand_ratio=6.0)
    x = universal_inverted_block(x, inp_channels=160, out_channels=160, start_dw_kernel=3, middle_dw_kernel=3,
                                 middle_dw_downsample=True, stride=1, expand_ratio=2.0)
    # Stage 4&#8203;:contentReference[oaicite:52]{index=52}&#8203;:contentReference[oaicite:53]{index=53}: UIB + MHSA blocks (interleaved)
    # Initial sequence of UIBs (4 blocks) before attention&#8203;:contentReference[oaicite:54]{index=54}:
    x = universal_inverted_block(x, inp_channels=160, out_channels=160, start_dw_kernel=3, middle_dw_kernel=5,
                                 middle_dw_downsample=True, stride=2, expand_ratio=6.0)  # downsample
    x = universal_inverted_block(x, inp_channels=160, out_channels=160, start_dw_kernel=0, middle_dw_kernel=0,
                                 middle_dw_downsample=True, stride=1, expand_ratio=2.0)
    x = universal_inverted_block(x, inp_channels=160, out_channels=160, start_dw_kernel=3, middle_dw_kernel=3,
                                 middle_dw_downsample=True, stride=1, expand_ratio=4.0)
    x = universal_inverted_block(x, inp_channels=160, out_channels=160, start_dw_kernel=3, middle_dw_kernel=5,
                                 middle_dw_downsample=True, stride=1, expand_ratio=4.0)
    # Now interleave attention blocks (each MHSA at 24x24 spatial, so kv_strides=2 as per px=24)&#8203;:contentReference[oaicite:55]{index=55}
    # Attention block 1 (after 4 UIBs)
    x = attention_block(x, num_heads=4, key_dim=64, value_dim=64, kv_strides=2, use_layer_scale=True, use_residual=True)
    # UIB block 5
    x = universal_inverted_block(x, inp_channels=160, out_channels=160, start_dw_kernel=3, middle_dw_kernel=3,
                                 middle_dw_downsample=True, stride=1, expand_ratio=4.0)
    # Attention block 2
    x = attention_block(x, num_heads=4, key_dim=64, value_dim=64, kv_strides=2, use_layer_scale=True, use_residual=True)
    # UIB block 6 (ConvNext variant)
    x = universal_inverted_block(x, inp_channels=160, out_channels=160, start_dw_kernel=3, middle_dw_kernel=0,
                                 middle_dw_downsample=True, stride=1, expand_ratio=4.0)
    # Attention block 3
    x = attention_block(x, num_heads=4, key_dim=64, value_dim=64, kv_strides=2, use_layer_scale=True, use_residual=True)
    # UIB block 7
    x = universal_inverted_block(x, inp_channels=160, out_channels=160, start_dw_kernel=3, middle_dw_kernel=3,
                                 middle_dw_downsample=True, stride=1, expand_ratio=4.0)
    # Attention block 4
    x = attention_block(x, num_heads=4, key_dim=64, value_dim=64, kv_strides=2, use_layer_scale=True, use_residual=True)
    # UIB block 8 (final of stage4)
    x = universal_inverted_block(x, inp_channels=160, out_channels=160, start_dw_kernel=3, middle_dw_kernel=0,
                                 middle_dw_downsample=True, stride=1, expand_ratio=4.0)
    # Stage 5&#8203;:contentReference[oaicite:56]{index=56}&#8203;:contentReference[oaicite:57]{index=57}: UIB + MHSA blocks (interleaved)
    # Initial sequence of UIBs (8 blocks) before attention:
    x = universal_inverted_block(x, inp_channels=160, out_channels=256, start_dw_kernel=5, middle_dw_kernel=5,
                                 middle_dw_downsample=True, stride=2, expand_ratio=6.0)  # downsample to 7x7
    x = universal_inverted_block(x, inp_channels=256, out_channels=256, start_dw_kernel=5, middle_dw_kernel=5,
                                 middle_dw_downsample=True, stride=1, expand_ratio=4.0)
    x = universal_inverted_block(x, inp_channels=256, out_channels=256, start_dw_kernel=3, middle_dw_kernel=5,
                                 middle_dw_downsample=True, stride=1, expand_ratio=4.0)
    x = universal_inverted_block(x, inp_channels=256, out_channels=256, start_dw_kernel=3, middle_dw_kernel=5,
                                 middle_dw_downsample=True, stride=1, expand_ratio=4.0)
    x = universal_inverted_block(x, inp_channels=256, out_channels=256, start_dw_kernel=0, middle_dw_kernel=0,
                                 middle_dw_downsample=True, stride=1, expand_ratio=2.0)
    x = universal_inverted_block(x, inp_channels=256, out_channels=256, start_dw_kernel=3, middle_dw_kernel=5,
                                 middle_dw_downsample=True, stride=1, expand_ratio=2.0)
    x = universal_inverted_block(x, inp_channels=256, out_channels=256, start_dw_kernel=0, middle_dw_kernel=0,
                                 middle_dw_downsample=True, stride=1, expand_ratio=2.0)
    x = universal_inverted_block(x, inp_channels=256, out_channels=256, start_dw_kernel=0, middle_dw_kernel=0,
                                 middle_dw_downsample=True, stride=1, expand_ratio=4.0)
    # Now interleave attention blocks at 12x12 spatial (kv_strides=1 for px=12)&#8203;:contentReference[oaicite:58]{index=58}
    x = attention_block(x, num_heads=4, key_dim=64, value_dim=64, kv_strides=1, use_layer_scale=True, use_residual=True)
    x = universal_inverted_block(x, inp_channels=256, out_channels=256, start_dw_kernel=3, middle_dw_kernel=0,
                                 middle_dw_downsample=True, stride=1, expand_ratio=4.0)
    x = attention_block(x, num_heads=4, key_dim=64, value_dim=64, kv_strides=1, use_layer_scale=True, use_residual=True)
    x = universal_inverted_block(x, inp_channels=256, out_channels=256, start_dw_kernel=5, middle_dw_kernel=5,
                                 middle_dw_downsample=True, stride=1, expand_ratio=4.0)
    x = attention_block(x, num_heads=4, key_dim=64, value_dim=64, kv_strides=1, use_layer_scale=True, use_residual=True)
    x = universal_inverted_block(x, inp_channels=256, out_channels=256, start_dw_kernel=5, middle_dw_kernel=0,
                                 middle_dw_downsample=True, stride=1, expand_ratio=4.0)
    x = attention_block(x, num_heads=4, key_dim=64, value_dim=64, kv_strides=1, use_layer_scale=True, use_residual=True)
    x = universal_inverted_block(x, inp_channels=256, out_channels=256, start_dw_kernel=5, middle_dw_kernel=0,
                                 middle_dw_downsample=True, stride=1, expand_ratio=4.0)
    # Final stage: 1x1 convs to 1280
    x = conv_bn_act(x, filters=960, kernel_size=1, strides=1, norm=True, act=True)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Reshape((1, 1, 960))(x)
    x = conv_bn_act(x, filters=1280, kernel_size=1, strides=1, norm=True, act=True)
    x = tf.keras.layers.Flatten()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs, name="MobileNetV4HybridMedium")

if __name__ == '__main__':
    # Test model creation
    model = MobileNetV4ConvSmall()
    model.summary()
    model.build((None, 224, 224, 3))
    test = tf.random.normal((1, 224, 224, 3))
    out = model(test)
    # model = MobileNetV4ConvMedium()
    # model.summary()
    # model = MobileNetV4ConvLarge()
    # model.summary()
    # model = MobileNetV4HybridMedium()
    # model.summary()