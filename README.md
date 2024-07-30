# Fashion MNIST Classification using Neural Network 
Classify apparel images in Fashion-MNIST dataset using custom built fully-connected neural network

<img src="https://github.com/sssingh/fashion-mnist-classification/blob/master/assets/fmnist_intro.png?raw=true" width="800" height="400">

## Features
⚡Multi Label Image Classification  
⚡Cutsom Fully Connected NN  
⚡Fashion MNIST  
⚡PyTorch

## Table of Contents
- [Introduction](#introduction) 
- [Objective](#objective)
- [Dataset](#dataset)
- [Evaluation Criteria](#evaluation-criteria)
- [Solution Approach](#solution-approach)
- [How To Use](#how-to-use)
- [License](#license)
- [Get in touch](#get-in-touch)

## Introduction
Just like [MNIST digit classification](https://github.com/sssingh/hand-written-digit-classification), the [Fashion-MNIST dataset](https://github.com/zalandoresearch/fashion-mnist) is a popular dataset for classification in the Machine Learning community for building and testing neural networks. MNIST is a pretty trivial dataset to be used with neural networks where one can quickly achieve better than 97% accuracy. Experts recommend ([Ian Goodfellow](https://twitter.com/goodfellow_ian/status/852591106655043584), [François Chollet](https://twitter.com/fchollet/status/852594987527045120)) to move away from MNIST dataset for model benchmarking and validation. Fashion-MNIST is more complex than MNIST, and it's a much better dataset for evaluating models than MNIST.

## Objective
We'll build a neural network using PyTorch. Only `fully-connected` layers will be used. The goal here is to classify ten classes of apparel images in the Fashion-MNIST dataset with as high accuracy as possible by only using fully-connected layers (i.e., without using `Convolution` layers)

## Dataset
- Dataset consists of 60,000 training images and 10,000 testing images.
- Every image in the dataset will belong to one of the ten classes...

| Label	| Description |
|--- | ---|
|0|	T-shirt/top|
|1|	Trouser|
|2|	Pullover|
|3|	Dress|
|4|	Coat|
|5|	Sandal|
|6|	Shirt|
|7|	Sneaker|
|8|	Bag|
|9|	Ankle boot|

- Each image in the dataset is a 28x28 pixel grayscale image, a zoomed-in single image shown below...

<img src="https://github.com/sssingh/fashion-mnist-classification/blob/master/assets/fmnist_single_image.png?raw=true">


- Here are zoomed-out samples of other images from the training dataset with their respective labels...

<img src="https://github.com/sssingh/fashion-mnist-classification/blob/master/assets/fmnist_samples.png?raw=true">


- We will use the in-built Fashion-MNIST dataset from PyTorch's `torchvision` package. The advantage of using the dataset this way is that we get a clean pre-processed dataset that pairs the image and respective label nicely, making our life easier when we iterate through the image samples while training and testing the model. Alternatively, the raw dataset can be downloaded from the original source. Like MNIST, the raw dataset comes as a set of 4 zip files containing training images, training image labels, testing images, and testing image labels in separate files... 
[train-images](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz), 
[train-labels](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz), 
[test-images](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz), 
[test-labels](http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz)
