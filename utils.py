# Helper functions for training and testing CNN as detailed in:
# 'Segmentation of histological images and fibrosis identification with a convolutional neural network'
# https://doi.org/10.1016/j.compbiomed.2018.05.015
# https://arxiv.org/abs/1803.07301

import numpy as np
import scipy as scp
import scipy.misc
import math
import tensorflow as tf

# Creates an image - colours correspond to each class
def generate_map(output):
	map = np.squeeze(output) # 4D -> 3D
	
	# no NaN
	if np.isnan(np.sum(map)):
		print('NaN in output map')
	# no tiny values
	if np.any(np.absolute(map)) < 0.0001:
		print('Very small values in output map')

	max_indexes = np.argmax(map, axis=2) # 2D matrix
	for i in range(map.shape[0]):
		for j in range(map.shape[1]):
		
			max_ind = max_indexes[i,j]
			assert(max_ind in [0,1,2])
			
			if max_ind == 0:		# red = myocyte
				map[i,j,0] = 150 
				map[i,j,1] = 0
				map[i,j,2] = 0
			elif max_ind == 1:		# white = background
				map[i,j,0] = 170
				map[i,j,1] = 180 
				map[i,j,2] = 175
			else:					# blue = fibrosis
				map[i,j,0] = 0
				map[i,j,1] = 0 
				map[i,j,2] = 180 
				
	return map

# Gets n unlabelled images in path with filename train_%d.png
def get_unlabelled(start_idx, batch_size, test=False):
	for i in range(start_idx, start_idx+batch_size):
		i += 1
		if test is True:
			img = scp.misc.imread("testing set/test_%d.png" %i)
		else:
			img = scp.misc.imread("training set/train_%d.png" %i)
		if i == int(start_idx + 1):
			img_h, img_w = img.shape[:2]
			unlabelled = np.array(img).reshape(1, img_h, img_w, 3)
		else:
			# Flatten, stack as 4d np array
			img = np.array(img).reshape(1, img_h, img_w, 3)
			unlabelled = np.concatenate((unlabelled, img), axis=0)

	assert(unlabelled.shape[0] == batch_size)
	
	# Apply preprocessing
	processed = preprocessing(unlabelled)
	return processed
	
# Mean subtraction & normalisation to unit variance per colour channel 
def preprocessing(unlabelled):
	unlabelled.astype(np.float64)
	means = [174.3182, 140.4974, 181.9621]	# RGB means of training set
	processed = unlabelled.astype(np.float64)
	for k in range(3):
		# Mean subtraction
		processed[:,:,:,k] = [x - means[k] for x in unlabelled[:,:,:,k]]
		for i in range(unlabelled.shape[0]):
			# Normalisation
			stddev = np.std(unlabelled[i,:,:,k], dtype=np.float64)
			processed[i,:,:,k] = [x / stddev for x in processed[i,:,:,k]]
	return processed
	
# Gets n ground truth images in path with filename train_%d_mask.png 
def get_labelled(start_idx, batch_size, test=False):
	for i in range(start_idx, start_idx+batch_size):
		i += 1
		if test is True:
			img = scp.misc.imread("testing set/test_%d_mask.png" %i)
		else:
			img = scp.misc.imread("training set/train_%d_mask.png" %i)
		if i == int(start_idx + 1):
			img_h, img_w = img.shape[:2]
			labelled = np.array(img).reshape(1, img_h, img_w, 3)
		else:
			# Flatten, stack as 4d np array
			img = np.array(img).reshape(1, img_h, img_w, 3)
			labelled = np.concatenate((labelled, img), axis=0)

	assert(labelled.shape[0] == batch_size)
	return labelled
	
# Computes accuracy and DSC of predictions
def compute_accuracy(pred, labels):

	# Find indexes of correct class and reshape to 1D
	pred_flat = np.argmax(pred, axis=3).flatten().astype(np.int64)
	labels_flat = np.argmax(labels, axis=3).flatten().astype(np.int64)
	assert(pred_flat.shape == labels_flat.shape)
	n_pixels = pred_flat.__len__()
	
	tally = np.zeros((3,3),dtype=np.int64)
	for i in range(n_pixels):
		a = labels_flat[i]
		b = pred_flat[i]
		tally[b,a] += 1		# Rows/down = predicted, columns/across = labels
	print(tally)
	
	pred_totals = np.sum(tally, axis=1)		# sum right 
	gt_totals = np.sum(tally, axis=0)		# sum down
	
	dsc = []
	for i in range(3):
		tp = tally[i,i]
		fp = pred_totals[i] - tally[i,i]
		fn = gt_totals[i] - tally[i,i]
		dsc.append(2*tp / (2*tp + fp + fn))
	avg_dsc = np.mean(dsc)
	accuracy = 100.0 * (tally[0,0] + tally[1,1] + tally[2,2]) / n_pixels
		
	return accuracy, avg_dsc