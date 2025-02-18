import tensorflow as tf

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

# Helper: Conv-BN-activation block
def conv_bn_relu(x, filters, kernel_size, strides=1, activation=True, name=None):
    x = tf.keras.layers.Conv2D(filters, kernel_size, strides=strides, padding='same', use_bias=False, name=name)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-3)(x)
    if activation:
        x = tf.keras.layers.ReLU()(x)
    return x

# Helper: Fused Inverted Bottleneck block (Conv + Conv instead of depthwise + conv)
def fused_inverted_bottleneck(x, in_filters, out_filters, kernel_size, strides, expand_ratio):
    # Expand
    expanded_filters = int(round(in_filters * expand_ratio))
    # 3x3 conv expansion (if expand_ratio=1, this is just to out_filters directly)
    x = tf.keras.layers.Conv2D(expanded_filters, kernel_size, strides=strides, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-3)(x)
    x = tf.keras.layers.ReLU()(x)
    # Project
    x = tf.keras.layers.Conv2D(out_filters, 1, strides=1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-3)(x)
    # Residual (only if stride=1 and in_filters == out_filters)
    if strides == 1 and in_filters == out_filters:
        x = x + 0  # (No residual here since strides not 1 in fused initial block of HybridMedium)
    return x

# Helper: Universal Inverted Bottleneck (UIB) block
def universal_inverted_bottleneck(x, in_filters, out_filters, expand_ratio,
                                  start_dw_kernel, middle_dw_kernel, strides, use_layer_scale=False):
    # Compute expanded channels
    expanded_filters = int(round(in_filters * expand_ratio))
    shortcut = x
    # Starting depthwise conv (if specified)
    if start_dw_kernel and start_dw_kernel > 0:
        # If a middle depthwise conv exists and we need to downsample, perform stride there (middle_dw_downsample=True)
        dw_stride = 1 if (middle_dw_kernel and middle_dw_kernel > 0 and strides > 1) else strides
        x = tf.keras.layers.DepthwiseConv2D(start_dw_kernel, strides=dw_stride, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-3)(x)
        # No activation after start depthwise (linear depthwise)&#8203;:contentReference[oaicite:32]{index=32}&#8203;:contentReference[oaicite:33]{index=33}
    # Expansion 1x1 conv
    x = tf.keras.layers.Conv2D(expanded_filters, 1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-3)(x)
    x = tf.keras.layers.ReLU()(x)
    # Middle depthwise conv (if specified)
    if middle_dw_kernel and middle_dw_kernel > 0:
        # If we deferred downsampling to middle DW (middle_dw_downsample=True), apply stride here
        dw_stride2 = strides if strides > 1 else 1
        x = tf.keras.layers.DepthwiseConv2D(middle_dw_kernel, strides=dw_stride2, padding='same', use_bias=False)(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-3)(x)
        x = tf.keras.layers.ReLU()(x)
    # Projection 1x1 conv
    x = tf.keras.layers.Conv2D(out_filters, 1, padding='same', use_bias=False)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-3)(x)
    # Optional Layer Scale (for UIB, often set False in this model&#8203;:contentReference[oaicite:34]{index=34}&#8203;:contentReference[oaicite:35]{index=35})
    if use_layer_scale:
        gamma = tf.Variable(1e-5 * tf.ones((out_filters,)), trainable=True)  # layer scale initialized to 1e-5
        x = gamma * x
    # Residual connection if shape matches and stride = 1
    if strides == 1 and in_filters == out_filters:
        x = tf.keras.layers.Add()([x, shortcut])
    return x

def MobileNetV4HybridMedium(input_shape=(224, 224, 3), num_classes=1000):
    inputs = tf.keras.Input(shape=input_shape)
    # Stage 1: initial conv 3x3, stride 2 -> 32 filters
    x = conv_bn_relu(inputs, filters=32, kernel_size=3, strides=2, activation=True, name='stem_conv')
    in_filters = 32
    # Stage 2: Fused inverted bottleneck: 3x3 conv stride 2 -> 48 filters
    x = fused_inverted_bottleneck(x, in_filters=in_filters, out_filters=48, kernel_size=3, strides=2, expand_ratio=4.0)
    in_filters = 48
    # Stage 3:
    # Block 3.1: UIB (start 3x3 dw, middle 5x5 dw) stride 2 -> 80 filters (expand 4.0)
    x = universal_inverted_bottleneck(x, in_filters=in_filters, out_filters=80, expand_ratio=4.0,
                                      start_dw_kernel=3, middle_dw_kernel=5, strides=2)
    in_filters = 80
    # Block 3.2: UIB (start 3x3 dw, middle 3x3 dw) stride 1 -> 80 filters (expand 2.0)
    x = universal_inverted_bottleneck(x, in_filters=in_filters, out_filters=80, expand_ratio=2.0,
                                      start_dw_kernel=3, middle_dw_kernel=3, strides=1)
    in_filters = 80  # output 80
    # Stage 4:
    # Block 4.1: UIB (start 3x3 dw, middle 5x5 dw) stride 2 -> 160 filters (expand 6.0)
    x = universal_inverted_bottleneck(x, in_filters=in_filters, out_filters=160, expand_ratio=6.0,
                                      start_dw_kernel=3, middle_dw_kernel=5, strides=2)
    in_filters = 160
    # Block 4.2: UIB (no depthwise) stride 1 -> 160 filters (expand 2.0). (This is an ExtraDW/FFN block with only 1x1 convs)
    x = universal_inverted_bottleneck(x, in_filters=in_filters, out_filters=160, expand_ratio=2.0,
                                      start_dw_kernel=0, middle_dw_kernel=0, strides=1)
    in_filters = 160
    # Block 4.3: UIB (start 3x3 dw, middle 3x3 dw) stride 1 -> 160 filters (expand 4.0)
    x = universal_inverted_bottleneck(x, in_filters=in_filters, out_filters=160, expand_ratio=4.0,
                                      start_dw_kernel=3, middle_dw_kernel=3, strides=1)
    in_filters = 160
    # Block 4.4: UIB (start 3x3 dw, middle 5x5 dw) stride 1 -> 160 filters (expand 4.0)
    x = universal_inverted_bottleneck(x, in_filters=in_filters, out_filters=160, expand_ratio=4.0,
                                      start_dw_kernel=3, middle_dw_kernel=5, strides=1)
    in_filters = 160
    # Block 4.5: MQA Self-Attention block at 14x14 (downsample K/V by 2 to 7x7)&#8203;:contentReference[oaicite:36]{index=36}&#8203;:contentReference[oaicite:37]{index=37}
    x = MultiHeadSelfAttentionBlock(input_dim=in_filters, output_dim=160, num_heads=4,
                                    key_dim=64, value_dim=64, use_multi_query=True,
                                    query_h_strides=1, query_w_strides=1, kv_strides=2,
                                    downsampling_dw_kernel_size=3, dropout=0.0,
                                    use_bias=False, use_layer_scale=True, use_residual=True)(x)
    # Block 4.6: UIB (start 3x3 dw, middle 3x3 dw) stride 1 -> 160 filters (expand 4.0)
    x = universal_inverted_bottleneck(x, in_filters=160, out_filters=160, expand_ratio=4.0,
                                      start_dw_kernel=3, middle_dw_kernel=3, strides=1)
    # Block 4.7: MQA Self-Attention (same 14x14 with K/V downsample) – repeated
    x = MultiHeadSelfAttentionBlock(input_dim=160, output_dim=160, num_heads=4,
                                    key_dim=64, value_dim=64, use_multi_query=True,
                                    query_h_strides=1, query_w_strides=1, kv_strides=2,
                                    downsampling_dw_kernel_size=3, dropout=0.0,
                                    use_bias=False, use_layer_scale=True, use_residual=True)(x)
    # Block 4.8: UIB (start 3x3 dw, no middle) stride 1 -> 160 filters (expand 4.0)
    x = universal_inverted_bottleneck(x, in_filters=160, out_filters=160, expand_ratio=4.0,
                                      start_dw_kernel=3, middle_dw_kernel=0, strides=1)
    # Block 4.9: MQA Self-Attention (14x14 with K/V downsample)
    x = MultiHeadSelfAttentionBlock(input_dim=160, output_dim=160, num_heads=4,
                                    key_dim=64, value_dim=64, use_multi_query=True,
                                    query_h_strides=1, query_w_strides=1, kv_strides=2,
                                    downsampling_dw_kernel_size=3, dropout=0.0,
                                    use_bias=False, use_layer_scale=True, use_residual=True)(x)
    # Block 4.10: UIB (start 3x3 dw, middle 3x3 dw) stride 1 -> 160 filters (expand 4.0)
    x = universal_inverted_bottleneck(x, in_filters=160, out_filters=160, expand_ratio=4.0,
                                      start_dw_kernel=3, middle_dw_kernel=3, strides=1)
    # Block 4.11: MQA Self-Attention (14x14 with K/V downsample)
    x = MultiHeadSelfAttentionBlock(input_dim=160, output_dim=160, num_heads=4,
                                    key_dim=64, value_dim=64, use_multi_query=True,
                                    query_h_strides=1, query_w_strides=1, kv_strides=2,
                                    downsampling_dw_kernel_size=3, dropout=0.0,
                                    use_bias=False, use_layer_scale=True, use_residual=True)(x)
    # Block 4.12: UIB (start 3x3 dw, no middle) stride 1 -> 160 filters (expand 4.0) [Output of stage4]
    x = universal_inverted_bottleneck(x, in_filters=160, out_filters=160, expand_ratio=4.0,
                                      start_dw_kernel=3, middle_dw_kernel=0, strides=1)
    in_filters = 160
    # Stage 5:
    # Block 5.1: UIB (start 5x5 dw, middle 5x5 dw) stride 2 -> 256 filters (expand 6.0)
    x = universal_inverted_bottleneck(x, in_filters=in_filters, out_filters=256, expand_ratio=6.0,
                                      start_dw_kernel=5, middle_dw_kernel=5, strides=2)
    in_filters = 256
    # Block 5.2: UIB (start 5x5 dw, middle 5x5 dw) stride 1 -> 256 filters (expand 4.0)
    x = universal_inverted_bottleneck(x, in_filters=in_filters, out_filters=256, expand_ratio=4.0,
                                      start_dw_kernel=5, middle_dw_kernel=5, strides=1)
    in_filters = 256
    # Block 5.3: UIB (start 3x3 dw, middle 5x5 dw) stride 1 -> 256 filters (expand 4.0)
    x = universal_inverted_bottleneck(x, in_filters=in_filters, out_filters=256, expand_ratio=4.0,
                                      start_dw_kernel=3, middle_dw_kernel=5, strides=1)
    in_filters = 256
    # Block 5.4: UIB (start 3x3 dw, middle 5x5 dw) stride 1 -> 256 filters (expand 4.0)
    x = universal_inverted_bottleneck(x, in_filters=in_filters, out_filters=256, expand_ratio=4.0,
                                      start_dw_kernel=3, middle_dw_kernel=5, strides=1)
    in_filters = 256
    # Block 5.5: UIB (no depthwise) stride 1 -> 256 filters (expand 2.0)
    x = universal_inverted_bottleneck(x, in_filters=in_filters, out_filters=256, expand_ratio=2.0,
                                      start_dw_kernel=0, middle_dw_kernel=0, strides=1)
    in_filters = 256
    # Block 5.6: UIB (start 3x3 dw, middle 5x5 dw) stride 1 -> 256 filters (expand 2.0)
    x = universal_inverted_bottleneck(x, in_filters=in_filters, out_filters=256, expand_ratio=2.0,
                                      start_dw_kernel=3, middle_dw_kernel=5, strides=1)
    in_filters = 256
    # Block 5.7: UIB (no depthwise) stride 1 -> 256 filters (expand 2.0)
    x = universal_inverted_bottleneck(x, in_filters=in_filters, out_filters=256, expand_ratio=2.0,
                                      start_dw_kernel=0, middle_dw_kernel=0, strides=1)
    in_filters = 256
    # Block 5.8: UIB (no depthwise) stride 1 -> 256 filters (expand 4.0)
    x = universal_inverted_bottleneck(x, in_filters=in_filters, out_filters=256, expand_ratio=4.0,
                                      start_dw_kernel=0, middle_dw_kernel=0, strides=1)
    in_filters = 256
    # Block 5.9: MQA Self-Attention block at 12x12 (no downsampling on K/V, kv_strides=1)&#8203;:contentReference[oaicite:38]{index=38}&#8203;:contentReference[oaicite:39]{index=39}
    x = MultiHeadSelfAttentionBlock(input_dim=in_filters, output_dim=256, num_heads=4,
                                    key_dim=64, value_dim=64, use_multi_query=True,
                                    query_h_strides=1, query_w_strides=1, kv_strides=1,
                                    downsampling_dw_kernel_size=3, dropout=0.0,
                                    use_bias=False, use_layer_scale=True, use_residual=True)(x)
    # Block 5.10: UIB (start 3x3 dw, no middle) stride 1 -> 256 filters (expand 4.0)
    x = universal_inverted_bottleneck(x, in_filters=256, out_filters=256, expand_ratio=4.0,
                                      start_dw_kernel=3, middle_dw_kernel=0, strides=1)
    # Block 5.11: MQA Self-Attention (12x12, kv_strides=1)
    x = MultiHeadSelfAttentionBlock(input_dim=256, output_dim=256, num_heads=4,
                                    key_dim=64, value_dim=64, use_multi_query=True,
                                    query_h_strides=1, query_w_strides=1, kv_strides=1,
                                    downsampling_dw_kernel_size=3, dropout=0.0,
                                    use_bias=False, use_layer_scale=True, use_residual=True)(x)
    # Block 5.12: UIB (start 5x5 dw, middle 5x5 dw) stride 1 -> 256 filters (expand 4.0)
    x = universal_inverted_bottleneck(x, in_filters=256, out_filters=256, expand_ratio=4.0,
                                      start_dw_kernel=5, middle_dw_kernel=5, strides=1)
    # Block 5.13: MQA Self-Attention (12x12, kv_strides=1)
    x = MultiHeadSelfAttentionBlock(input_dim=256, output_dim=256, num_heads=4,
                                    key_dim=64, value_dim=64, use_multi_query=True,
                                    query_h_strides=1, query_w_strides=1, kv_strides=1,
                                    downsampling_dw_kernel_size=3, dropout=0.0,
                                    use_bias=False, use_layer_scale=True, use_residual=True)(x)
    # Block 5.14: UIB (start 5x5 dw, no middle) stride 1 -> 256 filters (expand 4.0)
    x = universal_inverted_bottleneck(x, in_filters=256, out_filters=256, expand_ratio=4.0,
                                      start_dw_kernel=5, middle_dw_kernel=0, strides=1)
    # Block 5.15: MQA Self-Attention (12x12, kv_strides=1)
    x = MultiHeadSelfAttentionBlock(input_dim=256, output_dim=256, num_heads=4,
                                    key_dim=64, value_dim=64, use_multi_query=True,
                                    query_h_strides=1, query_w_strides=1, kv_strides=1,
                                    downsampling_dw_kernel_size=3, dropout=0.0,
                                    use_bias=False, use_layer_scale=True, use_residual=True)(x)
    # Block 5.16: UIB (start 5x5 dw, no middle) stride 1 -> 256 filters (expand 4.0) [Output of stage5]
    x = universal_inverted_bottleneck(x, in_filters=256, out_filters=256, expand_ratio=4.0,
                                      start_dw_kernel=5, middle_dw_kernel=0, strides=1)
    # Final stage: 1x1 conv to 960, global pooling, then 1x1 conv to 1280
    x = conv_bn_relu(x, filters=960, kernel_size=1, strides=1, activation=True, name='final_expand_conv')
    # Global average pooling (reduce spatial dimensions to 1x1)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)  # corresponds to 'gpooling'&#8203;:contentReference[oaicite:40]{index=40}
    # 1x1 conv to 1280 (embedding dimension before classifier)
    x = tf.keras.layers.Dense(1280, use_bias=False)(x)  # using Dense as 1x1 conv on pooled features
    x = tf.keras.layers.BatchNormalization(momentum=0.99, epsilon=1e-3)(x)
    x = tf.keras.layers.ReLU()(x)
    # Classification layer
    outputs = tf.keras.layers.Dense(num_classes, activation='softmax')(x)
    return tf.keras.Model(inputs, outputs, name="MobileNetV4-HybridMedium")

# Instantiate the model to ensure it's built
model = MobileNetV4HybridMedium()
