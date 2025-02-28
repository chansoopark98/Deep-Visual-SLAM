import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from model.flownet import Flownet
from flow_learner import FlowLearner
from util.metric import EndPointError
from util.plot import plot_images
from dataset.data_loader import DataLoader
from datetime import datetime
import tensorflow as tf, tf_keras
import keras
from tqdm import tqdm
import numpy as np
import yaml
import cv2

if __name__ == '__main__':
    with open('./flow/config.yaml', 'r') as file:
        config = yaml.safe_load(file)
    config['Train']['batch_size'] = 1

    saved_flownet_weights = './assets/weights/flow/epoch_100_model.weights.h5'

    with tf.device('/device:GPU:0'):
        model = Flownet(image_shape=(config['Train']['img_h'], config['Train']['img_w']),
                                batch_size=config['Train']['batch_size'],
                                prefix='flownet')
        model_input_shape = (config['Train']['batch_size'], config['Train']['img_h'], config['Train']['img_w'], 6)
        _ = model(tf.zeros(model_input_shape))
        model.build(input_shape=model_input_shape)
        model.load_weights(saved_flownet_weights)


    # read camera
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    