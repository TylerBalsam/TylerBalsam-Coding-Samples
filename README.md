# TylerBalsam-Coding-Samples

A demonstrating of my coding ability, in both TFLearn/Tensorflow (Python) and C++.

This repository currently contains 3 projects:

1. An autoencoder dual-bottleneck colorization network with a classification output.

2. A novel autoencoder dual-bottleneck colorization network with a binary classification output.

3. A class C++ project, implementing a version of the lower level floating point division operations.

Here's a brief overview of each project:

1. Classification Colorization:
NOTE: This folder contains the core pieces of the code from this project, as a demonstration only.

   This network is a dual bottlenecked classification network to colorize photos. The first part of the network is a typical autoencoder structure, and all of the outputs of the encoder layers are bridged to the equivalent decoder layer to help offload the flow of lower level details from the bottleneck. The secondary bottleneck is to reduce computation load and smooth the predicted classification distribution. The output layer depth is squeezed to the original layer depth (default is 8), then expanded to a quarter, then half, then full depth classification depth. The minimal bin size I found in practice to produce realistically colored images was 25 per dimension, which results in an output depth of 50 total for both Cb and Cr.
   The YCbCr colorspace is used for its simplicity. Loss is a combination of neg-ln and a total variance loss. One interesting fact to note is that due to the classification output, the total variance loss is applied to the confidence levels of the network, and in practice this effectively smoothed output colors well. It additionally functioned as a regularization method. 
	
#DESCRIPTION HERE

2. Binary Classification Colorization:
   NOTE: This folder the core pieces of the code from this project, as a demonstration only. 
#DESCRIPTION HERE

3. Floating Point Division:
#DESCRIPTION HERE
