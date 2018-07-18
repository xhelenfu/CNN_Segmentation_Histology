# Implementation to train the CNN as detailed in:
# 'Segmentation of histological images and fibrosis identification with a convolutional neural network'
# https://doi.org/10.1016/j.compbiomed.2018.05.015
# https://arxiv.org/abs/1803.07301

import os
import logging
import sys

import network

import tensorflow as tf
from tensorflow.python.framework import ops

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                    level=logging.INFO,
                    stream=sys.stdout)
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

batch_size = 128
n_train_data = 59904		# Number of RGB images
n_epochs = 100  				# Number of epochs to train for
train = True
restore = False			  	# Option to continue training from previous model
save = True			    		# Save model every epoch 
h = 48					      	# Image height
w = 48			      			# Image width
keep_rate = 1.0	  			# 1 - dropout rate

# Train neural network
logging.info("Training network")
convnet = network.CNN(keep_rate=keep_rate, train_mode=train)
t_net = network.TRAIN_CNN(convnet, batch_size, h, w)
t_net.train_network(n_train_data, batch_size, n_epochs, restore=restore, save=save)
