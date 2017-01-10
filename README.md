# TylerBalsam-Coding-Samples

A demonstrating of my coding ability, in both TFLearn/Tensorflow (Python) and C++. These are mostly coding snippets -- only the Floating Point Division program will run on its own.

This repository currently contains 3 projects:

1. An autoencoder dual-bottleneck colorization network with a classification output.

2. A novel autoencoder dual-bottleneck colorization network with a binary classification output.

3. A class C++ project, implementing a version of the lower level floating point division operations.

Here's a brief overview of each project:

## 1. Classification Colorization:

   This network is a dual bottlenecked classification network to colorize photos. The first part of the network is a typical autoencoder structure, and all of the outputs of the encoder layers are bridged to the equivalent decoder layer to help offload the flow of lower level details from the bottleneck. The secondary bottleneck is to reduce computation load and smooth the predicted classification distribution. The output layer depth is squeezed to the original layer depth (default is 8), then expanded to a quarter, then half, then full depth classification depth. The minimal bin size I found in practice to produce realistically colored images was 25 per dimension, which results in an output depth of 50 total for both Cb and Cr.

   The YCbCr colorspace is used for its simplicity. Loss is a combination of neg-ln and a total variance loss. One interesting fact to note is that due to the classification output, the total variance loss is applied to the confidence levels of the network, and in practice this effectively smoothed output colors well. It additionally functioned as a regularization method. Output confidences were split to their respective Cb and Cr counterparts, and softmaxed across the depth of each split at each pixel.


### Problems for consideration
   
   1. Currently, the value drawn from the forward pass of the colorization network is the value with the maximum confidence. This is because without some sort of object segmentation, colors drawn for a particular object from the predicted binned distribution would be speckled all over the map. Possible solutions include training a segmentation network on top of a pretrained version of this network and pulling a draw from the distribution similar across each segmented area. Due to the lower need for color resolution in the YCbCr colorspace, this would be a fairly fault tolerant. However, adapting a segmentation network to inputs of 224x224x25x2 could be extremely difficult/memory intensive.

   2. The network output, at 224x224x50 (which is converted to 224x224x25x2 for YCbCr binning), takes up a tremendous amount of memory in training and heavily limits batch size. Possible solutions include using lower layers of the VGG-16 network as a fixed encoder network, and train only the middle section and decoder network. With the encoder network no longer encumbered with the memory requirement of Adam, this allows for much larger batch sizes with moderately good performance. See results/VGG-16-Encoder-48-Hours for the uncurated validation set about 48 hours into training on a GTX 1070 using this method. An alternate solution would be to convert to an architecture similar to Conditional Adversarial Nets (see https://phillipi.github.io/pix2pix/). One solution I have implemented with promising results has been the below network, a Binary Classification Network.

	

## 2. Binary Classification Colorization:

   After a few months working with the above classification network, I decided to look for a better way to express its output. The main reason I had created the classification network was to separate the colors into distinct bins, thus solving the brown color issue that the Euclidean loss created. The Euclidean loss in an earlier network I used chose a brown color for many objects with a high color variance because it was averaging the colors it saw in practice -- a great mode of convergence for single-color objects like the grass and sky, but terrible for cars, dresses, balloons, and so on. One interesting property of a classification network was that it also showed promise of giving entire probability curve of the color distribution at that point. However, I quickly ran into problems with this methodology. One was that while you could plot the bin probabilities for the CB and CR dimensions on the x and y axis, and create a 3D "Probability Map" containing hills and valleys of probabilities for each bin at a 2D location, it was a lossy compression.
   
   By transcribing the peaks and the high points of the distribution, the valleys and smaller hills hidden "behind" the peaks in the 3D contour plot are lost -- leaving only guesses as to their real height. My solution to this was to only sample the highest point on each axis, which did achieve the effect of producing more vivid colors, but it discarded the rest of that information. One way to get around this would be to flatten every slice of that contour plot to 1D, and instead of plotting a 25x25 space at each point, plot a 625 bin plot at each point. However, even at a half scale of 112x112 pixels, this bin size would be almost pointless from a memory standpoint. So I took the thinking in the other direction, reasoning that if I was only going to draw the highest value, then I could effectively discard the rest of the data to entropy. Going from 1D to 2D had that effect, so I drew plans for a 4D draw, and higher, until I eventually ended up with binary, which resulted in an output of 2x6x2 bins, a pair of 1 and 0 bins for each of the 6 digits, and 1 set of digits for Cb and 1 for Cr. (Note: one could abstract it to a 1xNx2 system and flatten the CbCr bins to one dimension, but for simplicity's sake and POC I kept it separate.)

   A negative log likelihood loss is applied to the bins (softmaxed along the pair of 1 and 0 bins) during training time, and during the forward pass the bin with the highest confidence (i.e. > .5) is selected, then the binary digit at each point is converted to a color integer, which represents the Cb or Cr bin, respectively. Then that bin is converted to the lowest value in the bin, and stacked on the original BW image and saved to a .png file.

   It works surprisingly well, now that I have been able to work with it some. One adjustment I had previously made that caused issues was attempting to weight the digits' loss by their "influence" on the resultant base 10 number. This was effectively [32, 16, 8, 4, 2, 1] for a six digit binary output. However, disabling this caused the network to go from slow partial convergence on the sanity test to relatively quick convergence on the sanity test. While it does not converge quite as quickly as the normal classification network, which does not converge as quickly as a network with a Euclidean Loss, it has absolutely exceeded any expectations I could have had for it. I am currently working on optimizing it for convergence on a smaller test set, and initial results look promising. For the sanity test and the confidence map, see the relevant folder under colorization_binary. One thing to note is that the confidence map for this network on the sanity test, compared to the classification network, is more speckled. From my past experience, lowering speckling will be a combination of adjusting the total variance loss constant, the learning rate, and increasing the epoch numbers during training time. Running post-processing is also an option as well.


## 3. Floating Point Division:

   This was for a class project in the fall of 2015, and is a more vanilla demonstration of my coding ability. It is a relatively simple, lower level program to do floating point division on an input file of two floating point integers in hex, each pair separated by a space, one per line.
