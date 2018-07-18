# Implementation to test the CNN as detailed in:
# 'Segmentation of histological images and fibrosis identification with a convolutional neural network'
# https://doi.org/10.1016/j.compbiomed.2018.05.015
# https://arxiv.org/abs/1803.07301

# Test segmentation performance of the models which were saved at each epoch during training 
# Computes mean accuracy and DSC across test set for each model 

import numpy as np
import scipy as scp
import scipy.misc
import os
import logging
import sys

import network
import utils

import tensorflow as tf
from tensorflow.python.framework import ops

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                   level=logging.INFO,
                   stream=sys.stdout)
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

batch_size = 1          # 1 test image at a time
n_epochs = 2			# Number of models
train = False			
# restore = True
# save = False
h = 1536				# Image height 
w = 2064				# Image width 
img_idx = 0 			# train_(n-1).png in folder, n-1 = img_idx
n_predict = 48			# Number of test images
keep_rate = 1.0			# 1 - dropout rate

# Initialise model
logging.info("Getting predictions")
convnet = network.CNN(keep_rate=keep_rate, train_mode=train)
images = tf.placeholder(tf.float32, shape=(batch_size, h, w, 3))

# Build network
convnet.build(images)
logging.info("Finished building network")

# Get and save predictions
epoch_acc = np.zeros(n_epochs)
epoch_dsc = np.zeros(n_epochs)
for j in range(n_epochs):
	init = tf.global_variables_initializer()
		
	# if restore is True:
	saver = tf.train.Saver()

	with tf.Session() as sess:
		sess.run(init)
		
		# Reload current model
		saver.restore(sess, "model/epoch_%d/model.ckpt" %(j+1))
		logging.info("Model restored for prediction")
					
		for i in range(img_idx,(img_idx+n_predict)):
			
			# Get prediction for input image 
			print("Epoch %d, image %d of %d" %((j+1), (i+1), n_predict))
			unlabelled = utils.get_unlabelled(i, batch_size, test=True)
			pred = sess.run(convnet.out_max, feed_dict={images: unlabelled})

			# Compute accuracy and dsc if mask is available
			if os.path.isfile("testing set/test_%d_mask.png" %(i+1)):
				labels = utils.get_labelled(i, 1, test=True)
				accuracy, dsc = utils.compute_accuracy(pred, labels)
				print("Prediction percent accuracy: %.3f and DSC: %.3f" %(accuracy, dsc))
				epoch_acc[j] += accuracy
				epoch_dsc[j] += dsc
				
				logging.info("Creating output map")
				map = utils.generate_map(pred)
				scp.misc.imsave('pred testing/pred_%d_epoch_%d_a_%.3f_d_%.3f.png' 
								%(i+1, j+1, accuracy, dsc), map)
			else:
				print("Mask not found. Cannot compute accuracy and DSC")
				logging.info("Creating output map")
				map = utils.generate_map(pred)
				scp.misc.imsave('pred testing/pred_%d_epoch_%d.png' %(i+1, j+1), map)

# Stats for each epoch
epoch_acc = np.divide(epoch_acc, n_predict)
epoch_dsc = np.divide(epoch_dsc, n_predict)
print('Accuracy each epoch')
print(epoch_acc)
print('DSC each epoch')
print(epoch_dsc)
print('Best accuracy and DSC with epoch')
print(np.amax(epoch_acc), np.argmax(epoch_acc)+1, np.amax(epoch_dsc), np.argmax(epoch_dsc)+1)