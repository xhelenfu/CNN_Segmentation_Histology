# Implementation of CNN and training via minibatch gradient descent as detailed in:
# 'Segmentation of histological images and fibrosis identification with a convolutional neural network'
# https://doi.org/10.1016/j.compbiomed.2018.05.015
# https://arxiv.org/abs/1803.07301

import numpy as np
import scipy as scp
import tensorflow as tf
import os
import logging
import sys

import utils

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)

# Define CNN and functions for each layer
class CNN: 

	def __init__(self, keep_rate, train_mode=False):
		self.keep_rate = keep_rate
		self.train_mode = train_mode
		
	# Network architecture
	def build(self, img, y=None):
		
		logging.info("Started build")
		
		# Convert images to floating point RGB
		with tf.name_scope('Processing'):
			img = tf.image.convert_image_dtype(img, tf.float32)
			print("Input shape: %s" %(img.get_shape(),))
		
		# CNN
		self.conv1_1 = self.conv_layer(img, 3, 64, "conv1_1")
		self.conv1_2 = self.conv_layer(self.conv1_1, 64, 64, "conv1_2")
		self.conv1_3 = self.conv_layer(self.conv1_2, 64, 64, "conv1_3")
		self.conv1_4 = self.conv_layer(self.conv1_3, 64, 64, "conv1_4")
		self.conv1_5 = self.conv_layer(self.conv1_4, 64, 64, "conv1_5")
		self.conv1_6 = self.conv_layer(self.conv1_5, 64, 64, "conv1_6")
		self.conv1_7 = self.conv_layer(self.conv1_6, 64, 64, "conv1_7")
		self.conv1_8 = self.conv_layer(self.conv1_7, 64, 64, "conv1_8")
		self.conv1_9 = self.conv_layer(self.conv1_8, 64, 64, "conv1_9")
		
		self.conv2 = self.conv_layer(self.conv1_9, 64, 3, "conv2")
		self.conv3 = self.conv_layer(self.conv2, 3, 3, "conv3")
		
		self.out_max = self.pixel_wise_softmax(self.conv3)
		print("Network output shape: %s" %(self.out_max.get_shape(),))

		if self.train_mode:
			self.cost = self.compute_cost(self.conv3, y)
			self.accuracy = self.compute_accuracy_tensor(self.out_max, y)
	
	# Convolutional layer, filter 3x3 stride 1
	# ReLU and batch normalization 
	def conv_layer(self, x, in_depth, out_depth, name):
		with tf.variable_scope(name) as scope:
			filter = self.conv_filter(in_depth, out_depth, name)
			conv_2d = tf.nn.conv2d(x, filter, strides=[1, 1, 1, 1], padding='SAME')
			after_relu = tf.nn.relu(conv_2d)
			after_bn = self.batch_norm(after_relu, out_depth)
			
			print("Conv output shape: %s" %(after_bn.get_shape(),))
			return after_bn

	# Convolutional filter, truncated normal weight initialisation
	def conv_filter(self, in_depth, out_depth, name):
		cf_name = name + "_cf"
		
		if name is not ("conv2" or "conv3"):
			stddev = np.sqrt(2 / (9 * in_depth))
			filter = tf.Variable(tf.truncated_normal([3, 3, in_depth, out_depth], 
														dtype=tf.float32, stddev=stddev),
														name=cf_name)
		elif name is "conv2":
			stddev = np.sqrt(2 / 64)
			filter = tf.Variable(tf.truncated_normal([1, 1, 64, 3], dtype=tf.float32,
														stddev=stddev), name=cf_name)
		else:
			stddev = np.sqrt(2 / 3)
			filter = tf.Variable(tf.truncated_normal([1, 1, 3, 3], dtype=tf.float32,
														stddev=stddev), name=cf_name)
		return filter
		
	# Batch normalization, implementation based on:
	# https://gist.github.com/tomokishii/0ce3bdac1588b5cca9fa5fbdf6e1c412
	def batch_norm(self, x, out_depth):
		with tf.variable_scope('bn'):
			beta = tf.Variable(tf.constant(0.0, shape=[out_depth]), name='beta', trainable=True)
			gamma = tf.Variable(tf.constant(1.0, shape=[out_depth]), name='gamma', trainable=True)
			
			if self.train_mode is True:
				is_training = tf.constant(True)
			else:
				is_training = tf.constant(False)
			
			batch_mean, batch_var = tf.nn.moments(x, [0,1,2], name='moments')
			ema = tf.train.ExponentialMovingAverage(decay=0.9)

			def mean_var_with_update():
				ema_apply_op = ema.apply([batch_mean, batch_var])
				with tf.control_dependencies([ema_apply_op]):
					return tf.identity(batch_mean), tf.identity(batch_var)

			mean, var = tf.cond(is_training, mean_var_with_update, lambda: (ema.average(batch_mean), ema.average(batch_var)))
			bn = tf.nn.batch_normalization(x, mean, var, beta, gamma, 1e-3)
			
			return bn
					
	# Softmax = exp(logits) / reduce_sum(exp(logits), dim)
	def pixel_wise_softmax(self, logits):
		exp_logits = tf.exp(logits)
		sum_exp_logits = tf.reduce_sum(exp_logits, axis=3, keepdims=True)  # depth = 1
		sum_exp_logits_3 = tf.tile(sum_exp_logits, tf.stack([1, 1, 1, 3])) # depth = 3
		softmax = tf.div(exp_logits, sum_exp_logits_3)                     # element-wise division
		return softmax
		
	# tf.sparse_softmax_cross_entropy_with_logits -- computes softmax and cross entropy
	def compute_cost(self, logits, labels):
		# Reshapes labelled image to [batch_size, h, w], each entry is 0, 1, or 2 (int64) and 
		# corresponds to the depth layer of conv3 output map
		# Input shape [batch_size, h, w, 3]
		# R > (G, B) = cardiomyocyte = 0
		# G > (R, B) = background = 1
		# B > (R, G) = fibrotic region = 2
		labels_one_hot = tf.argmax(labels, axis=3)
		
		# logits shape [batch_size, h, w, 3], labels shape [batch_size, h, w] dtype=int64
		cost = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels_one_hot)
		return tf.reduce_mean(cost)
		
	# Computes accuracy of predictions, both inputs are tensors
	def compute_accuracy_tensor(self, pred, labels):
		correct = tf.equal(tf.argmax(pred, 3), tf.argmax(labels, 3))
		accuracy = 100 * tf.reduce_mean(tf.cast(correct, tf.float32))
		return accuracy
	
# CNN training 
class TRAIN_CNN(object):

	def __init__(self, cnn, batch_size, h, w):
		self.cnn = cnn
		self.x = tf.placeholder(tf.float32, shape=[batch_size, h, w, 3])
		self.y = tf.placeholder(tf.float32, shape=[batch_size, h, w, 3])
		self.cnn.build(self.x, self.y)
		self.batch_size = batch_size
		self.optimizer = tf.train.AdamOptimizer().minimize(self.cnn.cost)
		logging.info("Building CNN for training")

	def train_network(self, n_train_data, batch_size=1, n_epochs=1, restore=False, save=False):
		
		# n_iters = number of passes, each pass using [batch size] number of examples
		n_iters = int(n_train_data / batch_size)
		
		saver = tf.train.Saver(max_to_keep=100)
		init = tf.global_variables_initializer()
		
		# Create a summary to monitor cost per batch
		tf.summary.scalar("loss", self.cnn.cost)
		# Create a summary to monitor accuracy per batch
		tf.summary.scalar("accuracy", self.cnn.accuracy)
		# Merge all summaries into a single op
		merged_summary_op = tf.summary.merge_all()
			
		with tf.Session() as sess:
			sess.run(init)
		
			# Restore model if flag set to true, path in form "model/model.ckpt"
			if restore:
				saver.restore(sess, "model/model.ckpt")
				logging.info("Model restored for training")

			# op to write logs to Tensorboard
			summary_writer = tf.summary.FileWriter("logs", graph=sess.graph)
				
			logging.info("Optimisation Starting")
			
			# For shuffling batches during optimisation
			order = np.linspace(0, (n_iters-1)*batch_size, num=n_iters, dtype=np.int32)
			
			# Run optimiser for each batch over n_epochs
			for epoch in range(n_epochs):
				epoch_loss = 0
				epoch_acc = 0
					
				# Run optimiser for each batch of unlabelled and labelled 
				batch_iter = 0
				np.random.shuffle(order[:-1]) # Shuffle all but last batch
				for step in range((epoch*n_iters), ((epoch+1)*n_iters)):
					start_idx = order[batch_iter]
					epoch_x = utils.get_unlabelled(start_idx, batch_size)
					epoch_y = utils.get_labelled(start_idx, batch_size)
					_, c, b_acc, summary = sess.run([self.optimizer, self.cnn.cost, 
															self.cnn.accuracy,  
															merged_summary_op], 
															feed_dict={self.x: epoch_x, self.y: epoch_y})
					epoch_loss += c
					epoch_acc += b_acc
					batch_iter += 1
					
					# Write logs at every iteration
					summary_writer.add_summary(summary, step)

				# Compute and display DSC for this batch
				pred = sess.run(self.cnn.out_max, feed_dict={self.x: epoch_x})
				dsc = self.compute_dsc(pred, epoch_y)

				print("Epoch %d/%d with epoch loss: %.5f, epoch ACC: %.4f, last batch DSC: %.4f" 
						%(epoch+1, n_epochs, epoch_loss, epoch_acc/n_iters, dsc))

				# Save prediction for first image of last batch in current epoch
				self.save_prediction(pred[0,:,:,:], epoch, start_idx)
				
				# # Save updated model FOR LAST EPOCH ONLY
				# if save and epoch == (n_epochs-1):
					# save_path = saver.save(sess, "model/model.ckpt")
					# logging.info("Model saved in file: %s" % save_path)
						
				if save:
					if not os.path.exists("model/epoch_%d" %(epoch+1)):
						os.makedirs("model/epoch_%d" %(epoch+1))
					save_path = saver.save(sess, "model/epoch_%d/model.ckpt" %(epoch+1))
					logging.info("Model saved in file: %s" % save_path)
				
			logging.info("Optimisation Complete")
	
	# Save prediction for each epoch as image
	def save_prediction(self, pred, epoch, start_idx):
		map = utils.generate_map(pred)
		scp.misc.imsave('predictions training/epoch_%d_img_%d.png' %(epoch+1, start_idx+1), map)		
	
	def compute_dsc(self, pred, labels):
		pred_flat = np.argmax(pred, axis=3).flatten().astype(np.int64)
		labels_flat = np.argmax(labels, axis=3).flatten().astype(np.int64)
		assert(pred_flat.shape == labels_flat.shape)
		n_pixels = pred_flat.__len__()

		tally = np.zeros((3,3),dtype=np.int64)
		for i in range(n_pixels):
			a = labels_flat[i]
			b = pred_flat[i] 
			tally[b,a] += 1    # Rows/down = predicted, columns/across = labels
		print(tally)
		
		pred_totals = np.sum(tally, axis=1)
		gt_totals = np.sum(tally, axis=0)
		
		dsc = []
		for i in range(3):
			tp = tally[i,i]
			fp = pred_totals[i] - tally[i,i]
			fn = gt_totals[i] - tally[i,i]
			dsc.append(2*tp / (2*tp + fp + fn))
		avg_dsc = np.mean(dsc)
			
		return avg_dsc