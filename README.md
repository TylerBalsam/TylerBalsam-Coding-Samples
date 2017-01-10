# TylerBalsam-Coding-Samples

A demonstrating of my coding ability, in both TFLearn/Tensorflow (Python) and C++.

This repository currently contains 3 projects:

1. An autoencoder dual-bottleneck colorization network with a classification output.

2. A novel autoencoder dual-bottleneck colorization network with a binary classification output.

3. A class C++ project, implementing a version of the lower level floating point division operations.

Here's a brief overview of each project:

##1. Classification Colorization:

   ###NOTE: This folder contains the core pieces of the code from this project, as a demonstration only.

   This network is a dual bottlenecked classification network to colorize photos. The first part of the network is a typical autoencoder structure, and all of the outputs of the encoder layers are bridged to the equivalent decoder layer to help offload the flow of lower level details from the bottleneck. The secondary bottleneck is to reduce computation load and smooth the predicted classification distribution. The output layer depth is squeezed to the original layer depth (default is 8), then expanded to a quarter, then half, then full depth classification depth. The minimal bin size I found in practice to produce realistically colored images was 25 per dimension, which results in an output depth of 50 total for both Cb and Cr.

   The YCbCr colorspace is used for its simplicity. Loss is a combination of neg-ln and a total variance loss. One interesting fact to note is that due to the classification output, the total variance loss is applied to the confidence levels of the network, and in practice this effectively smoothed output colors well. It additionally functioned as a regularization method. Output confidences were split to their respective Cb and Cr counterparts, and softmaxed across the depth of each split at each pixel. These confidences are then squared -- in theory this gives extra precedence to rarer colors in the dataset, and in practice I've found it converges to more visually pleasing results more quickly.


   ##Problems for consideration
   
   1. Currently, the value drawn from the forward pass of the colorization network is the value with the maximum confidence. This is because without some sort of object segmentation, colors drawn for a particular object from the predicted binned distribution would be speckled all over the map. Possible solutions include training a segmentation network on top of a pretrained version of this network and pulling a draw from the distribution similar across each segmented area. Due to the lower need for color resolution in the YCbCr colorspace, this would be a fairly fault tolerant. However, adapting a segmentation network to inputs of 224x224x25x2 could be extremely difficult/memory intensive.

   2. The network output, at 224x224x50 (which is converted to 224x224x25x2 for YCbCr binning), takes up a tremendous amount of memory in training and heavily limits batch size. Possible solutions include using lower layers of the VGG-16 network as a fixed encoder network, and train only the middle section and decoder network. With the encoder network no longer encumbered with the memory requirement of Adam, this allows for much larger batch sizes with moderately good performance. See results/VGG-16-Encoder-48-Hours for the uncurated validation set about 48 hours into training on a GTX 1070 using this method. An alternate solution would be to convert to an architecture similar to Conditional Adversarial Nets (see https://phillipi.github.io/pix2pix/)

	

##2. Binary Classification Colorization:

   ##NOTE: This folder the core pieces of the code from this project, as a demonstration only. 

   This network is similar to the above network, with one major exception: The output network is 

   Potential convergence issue: Even with weight rebalancing, if the first node is incorrect, if the sub-node is correct then it will go further from the answer. So the question would be whether rebalancing the loss traversal to emphasize nodes with a lesser physical distance (like a euclidian loss for binary tree point distances), or whether a weighed independant weighted xentropy of the predicted/true digits is better. I could see the xentropy version fluctuating wildly for the first digit at first, then settling on lower digits as the network neared convergence. Proof is far away for that. PERHAPS, perhaps, the loss could ignore all lower nodes after the first incorrect digit? This could dampen a lot of the flip flopping noise, keep the high loss at high values, and allow more safely for that "settling" action. Binary Tree at: http://archive.cnx.org/contents/cffc6b16-b811-4c86-a4fd-ca7eae17735d@7/binary-codes-from-symbols-to-binary-codes

   Description here.

##3. Floating Point Division:

   This was for a class project in the fall of 2015, and is a more vanilla demonstration of my coding ability. It is a relatively simple, lower level program to do floating point division on an input file of two floating point integers in hex, one pair separated by a space per line.

   Further description here.
