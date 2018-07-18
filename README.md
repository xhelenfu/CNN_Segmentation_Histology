# Segmentation of histological images and fibrosis identification with a convolutional neural network

This repository contains an implementation of the convolutional neural network (CNN) described in this [paper](https://arxiv.org/ftp/arxiv/papers/1803/1803.07301.pdf).
The CNN segments biomedical images by predicting the class of each pixel.
In the original task, the images were of cardiac histological sections stained with Masson's trichrome. The objective was to segment the RGB images into 3 classes: myocyte, background, or fibrosis. 

![alt text](https://ars.els-cdn.com/content/image/1-s2.0-S0010482518301288-fx1_lrg.jpg "CNN Fu et al")

The code requires Python and TensorFlow to run.

## Training
Training data should be in the `./training set` folder. RGB images should be named in the format `train_1.png`, `train_2.png`, etc. Ground truth masks should be named in the format `train_1_mask.png`, `train_2_mask.png`, etc.

In `train.py`, ensure the variables in the list from `batch_size` to `keep_rate` are correct. 

To train the model, run `train.py`

A model is saved every epoch in the `./model` folder, TensorBoard logs are saved under `./logs`. One prediction is saved per epoch under `./predictions training`.

## Testing
Test data should be in the `./testing set` folder. RGB images should be named in the format `test_1.png`, `test_2.png`, etc. If available, ground truth masks should be named in the format `test_1_mask.png`, `test_2_mask.png`, etc. 

In `test.py`, ensure the variables in the list from `n_epochs` to `n_predict` are correct. 

To test, run `test.py`

Predicted segmentations are saved under `./predictions test`.

Accuracy and DSC scores are computed for each test image.
