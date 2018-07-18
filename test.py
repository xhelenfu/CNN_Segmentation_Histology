# Implementation to test the CNN as detailed in:
# 'Segmentation of histological images and fibrosis identification with a convolutional neural network'
# https://doi.org/10.1016/j.compbiomed.2018.05.015
# https://arxiv.org/abs/1803.07301

# Test segmentation performance of the models which were saved at each epoch during training 
# Computes mean accuracy and DSC across test set for each model 

import numpy as np
import scipy as scp
import tensorflow as tf
import os
import logging
import sys

import network
import utils

logging.basicConfig(format='%(asctime)s %(levelname)s %(message)s',
                   level=logging.INFO,
                   stream=sys.stdout)
# os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

n_epochs = 100  # Number of models
h = 1536        # Image height 
w = 2064        # Image width 
img_idx = 0     # test_(n-1).png in folder, n-1 = img_idx
n_predict = 48  # Number of test images

if not os.path.exists("predictions test"):
    os.makedirs("predictions test")

# Initialise model
logging.info("Getting predictions")
convnet = network.CNN(keep_rate=1.0, train_mode=False)
images = tf.placeholder(tf.float32, shape=(1, h, w, 3))

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
			unlabelled = utils.get_unlabelled(i, batch_size=1, test=True)
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
				scp.misc.imsave('predictions test/pred_%d_epoch_%d_a_%.3f_d_%.3f.png' 
								%(i+1, j+1, accuracy, dsc), map)
			else:
				print("Mask not found. Cannot compute accuracy and DSC")
				logging.info("Creating output map")
				map = utils.generate_map(pred)
				scp.misc.imsave('predictions test/pred_%d_epoch_%d.png' %(i+1, j+1), map)

# Stats for each epoch
epoch_acc = np.divide(epoch_acc, n_predict)
epoch_dsc = np.divide(epoch_dsc, n_predict)
print('Accuracy each epoch')
print(epoch_acc)
print('DSC each epoch')
print(epoch_dsc)
print('Best accuracy and DSC with epoch')
print(np.amax(epoch_acc), np.argmax(epoch_acc)+1, np.amax(epoch_dsc), np.argmax(epoch_dsc)+1)