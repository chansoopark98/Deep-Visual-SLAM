import tensorflow as tf

ssim_ratio = 0.85

def compute_reprojection_loss(reproj_image, tgt_image):
    l1_loss = tf.reduce_mean(tf.abs(reproj_image-tgt_image), axis=3, keepdims=True)

    ssim_loss = tf.reduce_mean(SSIM(reproj_image, tgt_image), axis=3, keepdims=True)

    loss = ssim_ratio * ssim_loss + (1 - ssim_ratio) * l1_loss
    #loss = l1_loss

    return loss

def SSIM(x, y):
    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    x = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')
    y = tf.pad(y, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

    mu_x = tf.nn.avg_pool2d(x, 3, 1, 'VALID')
    mu_y = tf.nn.avg_pool2d(y, 3, 1, 'VALID')

    sigma_x = tf.nn.avg_pool2d(x ** 2, 3, 1, 'VALID') - mu_x ** 2
    sigma_y = tf.nn.avg_pool2d(y ** 2, 3, 1, 'VALID') - mu_y ** 2
    sigma_xy = tf.nn.avg_pool2d(x * y, 3, 1, 'VALID') - mu_x * mu_y

    SSIM_n = (2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)
    SSIM_d = (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x + sigma_y + C2)

    SSIM = SSIM_n / SSIM_d

    return tf.clip_by_value((1 - SSIM) / 2, 0, 1)

def get_smooth_loss(disp, img):
    norm_disp = disp / ( tf.reduce_mean(disp, [1, 2], keepdims=True) + 1e-7)

    grad_disp_x = tf.abs(norm_disp[:, :-1, :, :] - norm_disp[:, 1:, :, :])
    grad_disp_y = tf.abs(norm_disp[:, :, :-1, :] - norm_disp[:, :, 1:, :])

    grad_img_x = tf.abs(img[:, :-1, :, :] - img[:, 1:, :, :])
    grad_img_y = tf.abs(img[:, :, :-1, :] - img[:, :, 1:, :])

    weight_x = tf.exp(-tf.reduce_mean(grad_img_x, 3, keepdims=True))
    weight_y = tf.exp(-tf.reduce_mean(grad_img_y, 3, keepdims=True))

    smoothness_x = grad_disp_x * weight_x
    smoothness_y = grad_disp_y * weight_y

    return tf.reduce_mean(smoothness_x) + tf.reduce_mean(smoothness_y)