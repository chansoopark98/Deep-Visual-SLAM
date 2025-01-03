import tensorflow as tf

if __name__ == '__main__':

    with tf.device('/GPU:1'):
        a = tf.keras.applications.EfficientNetV2S(include_top=False, weights='imagenet', input_shape=(224, 224, 3),
                                            classes=0, classifier_activation=None, include_preprocessing=True)
        
        a.summary()
    