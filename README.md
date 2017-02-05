# TylerBalsam-Coding-Samples

A demonstrating of my coding ability, in both TFLearn/Tensorflow (Python) and C++. These are mostly coding snippets -- only the Floating Point Division program will run on its own.

This repository currently contains 3 projects:

1. A bridged autoencoder-style dual-bottleneck colorization network with a classification output.

2. A novel bridged autoencoder-style dual-bottleneck colorization network with a binary classification output.

3. A C++ project for a previous programming class, implementing a version of the lower level floating point division operations.

The two neural network projects are the result of my past year of work in this area. Here's a overview of each project:

## 1. Classification Colorization:

   This network is a dual bottlenecked classification network to colorize photos. The first part of the network is a typical autoencoder structure, with all of the outputs of each of the encoder blocks bridged to the equivalent decoder block to help offload the flow of lower level details from the bottleneck. The secondary bottleneck is to reduce computation load and smooth the predicted classification distribution, since the output is a large classification output. In the secondary bottleneck, the output layer depth is squeezed to the original layer depth (default is 8), then expanded to a quarter, then half, then full depth classification depth. The minimal bin size I found in practice to produce realistically colored images was 25 per dimension, which results in an output depth of 50 total for both Cb and Cr, from the YCbCr colorspace.

   The YCbCr colorspace is used for its simplicity. Loss is a combination of neg-ln and a total variance loss. One interesting point to note is that due to the classification output, the total variance loss is applied to the confidence levels of the network, and in practice this effectively smoothed the entire color output distribution well (See Cool Trick #1 for more info). It additionally functioned as a regularization method. Output confidences were split to their respective Cb and Cr counterparts, and softmaxed across the depth of each split at each pixel location.

### A few of the cool tricks I used:
   
   1. Confidence smoothing: This one comes first as it is a personal favorite of mine. I found this as a solution to smoothing challenges for the classification network. Early on in the project I used a regression network with a MSE loss, and experienced quite a few problems with it. The primary one was the averaging issue that the MSE tends to cause. So I began building a classification output network to overcome the averaging issue. However, while you could directly smooth the real valued output of a regression network, adding a loss function that directly smoothed the real valued output of a classification network and was differentiable was not possible. This is because the function that samples the real values of the classification bins at testing time, ArgMax, is not differentiable, making it suited to a testing time forward pass but not for training, since it breaks the chain of differentiable functions. In training, as with most classification networks, the neg-ln loss on the confidence for the correct output bin was what I used as the loss instead. So the main values that were directly acessible were the confidences (See Cool Trick #4 for more details), and these came without any identifier of what bin location they were sourced from.  After some time spent working on this, I realized that many objects with a particular color or set of colors will often maintain that color consistently throughout, often only varying in brightness. Because brightness was exclusively contained in the Y channel (the black and white channel, or luminosity channel), then the Cb and Cr channels for most objects of a single color would be uniform across, and the actual color shade would become a more natural color gradient when the orginal black and white information was readded. See https://upload.wikimedia.org/wikipedia/commons/thumb/d/d9/Barns_grand_tetons_YCbCr_separation.jpg/800px-Barns_grand_tetons_YCbCr_separation.jpg for an example of this. Because of the relative consistency of encoded values in the Cb and Cr space, I reasoned that the shape probability distributions of each color at that point should be similar as well, and so therefore most slices, if not all, should be similar in shape barring some edge cases. With those things in mind, I reasoned that I could effectively increase generalization by applying total variance smoothing across the confidence levels. One point of reasoning behind this is as the confidence values come in to the loss function, distribution slices in physical proximity are likely to be from the same distribution, and should be similar, and thus their confidence levels should be similar as well. So the distribution's confidences are smoothed the same way as the distribution's shape is built: sample by sample. Furthermore, the Cb and Cr channels are a strong candidate for this type of smoothing because the sheer number of values that are similar to their neighbors in the Cb and Cr channels often outweighs the number of edges in those values, as the hard edges often lie in the Y channel, which we already have the data for. In practice, it this trick worked well. I found that small weighing values for this smoothing loss, like 1e-3, were more than sufficient for this particular network.

   2. Chroma Subsampling: Chroma subsampling was one trick that has been very effectively used for web image compression, video compression and more. It is an essentially "free" optimization. The core concept is that our eyes are less sensitive to color resolution than luminosity resolution in images (see https://upload.wikimedia.org/wikipedia/commons/0/06/Colorcomp.jpg for more details). In earlier tests of a less memory-intensive form of this network (using a pretrained VGG-16 network as an encoder network), I found that I was able to reduce the color resolution without any blatant issues. The current variation of the classification network in this repository does not use a reduced color resolution for the sake of modularity and personal experimentation, but past experiments have shown strong potential for greatly reducing the output size of the network using this as a later optimization during network design. The 48 hour VGG validation set photos (approximately 48 hours into training on a GTX 1070) in the /results folder demonstrates the VGG base variant of this network, which did use chroma subsampling. That particular folder is very early on in the network's training, so it is unfortunately only a potential demonstrator ability, and not full capacity.

   3. Subpixel Convolutions: Instead of using transposed convolutions, which are a sparse method of information generation, I opted to use subpixel covolutional layers after the first bottleneck (See https://github.com/Tetrachrome/subpixel for more info). Something I noted was that the Tetrachrome code seemed to function in a similar fashion to the tf.batch_to_space() operation with the channel and batch columns swapped, and so I implemented that with a block size of 2. It appeared to work effectively in place of transposed convolutions, and also appeared to almost completely remove the "checkerboard" subpatterns often output by networks that use transposed convolutions. One area I have yet to explore regarding this change that I have yet to experiment with is the number of of filters used for the convolution directly before the subpixel convolution layer. As I have not much experimented much with those parameters, I see a potential for results-to-parameters improvements there. One reason is because the subpixel convolutions reduce the depth of the incoming layer stack by a factor of four, and thus the layer bridge from the encoder network that gets concatenated to the output stack of the incoming output from the decoder network is actually taller than the generated output of the subpixel convolution layer! In a potentional worst case scenario where incoming information from the lower layers of the encoder network is useless, the decoder can ignore those layers and shift its weights to focus only on the layers coming off of the decoder stack. I'm not yet sold on the merits of repeatedly bottlenecking each decoder's depth, however.

   4. One-hot Masking: Where y_true is the one_hot representation of ground truths and y_pred is the softmaxed predicted distribution, instead of multiplying reduce_mean(reduce_sum(y_true\*log(y_pred), [1]), the equivalent log(boolean_mask(y_true, to_bool(y_pred))) was used. This allowed more flexibility for experimentation because it condensed a 224x224x2xn representation (for n classes) to a 224x224x2 output vector of the confidences at the y_true bin location. This opens the door to other fun operations on the data, such as outputting a heatmap of confidences at test time.

### Problems for consideration
   
   1. Currently, the value drawn from the forward pass of the colorization network is the value with the maximum confidence. This is because without some sort of object segmentation, colors drawn for a particular object from the predicted binned distribution would be speckled all over the place, and possibly incorrect since a 3D topographical reconstruction of the distribution would be from 2 2D representations, which presents an information loss scenario. One possible solution for this includes training a segmentation network on top of a pretrained version of this network and pulling a draw from the similar distributions across each segmented area. Due to the lower need for color resolution in the YCbCr colorspace, this should be a fairly fault tolerant approach. However, adapting a segmentation network to inputs of 224x224x25x2 could be extremely difficult/memory intensive, and I believe there are better solutions out there.

   2. The network output, at 224x224x50 (which is converted to 224x224x25x2 for YCbCr binning), takes up a tremendous amount of memory in training and heavily limits batch size. Possible solutions include the aforementioned chroma subsampling and/or using lower layers of the VGG-16 network as a fixed encoder network and training only a middle bottleneck section and a decoder network. With the encoder network no longer encumbered with the memory requirement of the gradients for traiing, this allows for much larger batch sizes with moderately good performance. See results/VGG-16-Encoder-48-Hours for the uncurated validation set about 48 hours into training on a GTX 1070 with a batch size of 50 using this method. An alternate solution would be to convert to an architecture similar to Conditional Adversarial Nets (see https://phillipi.github.io/pix2pix/) or StackGAN (see Item #4 for more info). One solution I have implemented with promising results has been the below network, a Binary Classification Network. I believe it deserves to stand on its own due to its novelty.

## 2. Binary Classification Colorization:

   After a few months working with the above classification network, I decided to look for a better way to express its output. As I've mentioned above, the main reason I had created the classification network was to separate the colors into distinct bins, thus solving the mean color issue that the regression network had. The MSE loss in an earlier network I used chose a brown color for many objects with a high color variance because it was approximating the posterior distribution by averaging the samples recieved -- a great mode of convergence for single-color objects like the grass and sky, but terrible for cars, dresses, balloons, and so on. One interesting property of the classification network that intrigued me early on was that it projected an estimate the posterior color distribution at each output pixel location. However, I quickly ran into problems implementing this method. One was that while you could plot the bin probabilities for the Cb and Cr dimensions on the x and y axis and thus reconstruct a 3D probability distribution containing the hills and valleys of probabilities for each bin at a 2D location, it was a lossy compression. Here's why:
   
   By transcribing the peaks and the high points of the distribution from two different perspectives in 2D (the Cb and Cr plots), the valleys and smaller hills obscured by the peaks in the 3D contour plot are lost -- leaving only guesses as to their real height and position. My solution to this was to only sample the highest point on each axis, which we can know is the most correctly projected after cross-entropy convergence. Implementing this method did achieve the effect of producing more distinct colors, but it also effectively discarded the rest of that color information. One way to get around this would be to flatten slices of that 3D contour plot to a long chain of 2D binned estimations at each point, and instead of plotting a 25x25 space of bins at each point, plot a 625 bin plot at each point. However, even at a half scale of 112x112 pixels with chroma subsampling, this bin size would be almost pointless from a memory-to-value standpoint with the limited resources I have, not to mention the potential problems of convergence speed and training set memorization due to increased parameter counts. As I developed this network, I took the thought process in the other direction, reasoning that if I was only going to draw the highest value from the distribution each time, and I could not feasibly represent the distribution a lower, more explicit set of dimensions, then the only way I could go (figuratively speaking) was up.

   In effect, I could effectively discard the rest of the data, those lower peaks and valleys, and write it off as lost information. In a higher dimensional representation of the probability distribution, that information would still be encoded in the network, but with the further-limited slice counts, disentangling these representations would be even more difficult. Going from 1D to 2D had that effect of effectively compressing the high peaks' expressed data at the cost of increased entropy. I reasoned that I could take a draw of the data in 4 rows of length N, which each mapped to the originally predicted bin numbers on the Cb and Cr planes. For example, for the Cb plane, a draw from the 4D distribution of [[4,2],[1,5]] would translate to bin 22 out of 25 in the Cb space, and bin 5 out of 25 in the CR space. Extending the thought process of abstraction, I kept raising the dimensions and shortening the sequence until I eventually ended up with N rows of length 2, which was of course a binary representation of the Cb & Cr embeddings. In practice, this resulted in an output of 2x6x2 bins, a pair of 1 and 0 bins for each of the 6 digits representing the chosen bin granularity (63 in this case), and 1 set of digits for Cb bin number and 1 for the Cr bin number. (Note: one could abstract it to a 1x(N\*2) system and flatten the CbCr bins to one dimension, but for simplicity's sake in implementing and testing the concept I kept them separate).

   A negative log likelihood loss is applied to the bins (softmaxed along each pair of 1 and 0 bins) during training time, and during the forward pass the bin with the highest confidence (i.e. > .5) is selected. In the forward pass, the binary representation at each point is converted to an integer which represents the Cb or Cr bin, respectively. Then that bin is mapped to the lowest real color value in the bin (this could also map to the midpoint, but past a certain color granularity it makes little difference: see Cool Trick #4 for more info), and stacked on the original BW image layer, converted and saved to a .png file.

   While it does not converge nearly as quickly as the normal classification network, which did not converge as quickly as a regression network, it has exceeded any expectations I could have had for it. After some modifications to the original design, I was able to get it to converge well on the sanity check, which is training on a single image to roughly guage saturation ability and convergence speed. While good performance on the sanity check doesn't guarantee good performance on larger data sets, using that sanity check has saved me hundreds of potentially wasted training hours for poor network designs. I am currently working on optimizing it for convergence on a smaller test set, and I'm seeing indications of convergence (albeit much more slowly). For the sanity check and the confidence map, see the relevant folder under colorization_binary. One thing to note is that the confidence map for this network on the sanity test, in comparison to the classification network, is more speckled. This could be either an indicator of a more fragile generalization, or just a side effect of the sanity check, since the sanity check tests the network convergence on just one image. From my past experience, lowering speckling in the future will be a combination of adjusting the total variance loss constant, the learning rate, and increasing the epoch numbers during training time. Running post-processing on the confidence value distribution is also an option as well.

   There is one potential failure case that I can see for this type of network. It requires thinking of the correct derivation of the output binary number as a binary tree with a 0-valued left leaf (with some underlying probability as a branch weight between any node and its leaf) and a 1-valued right leaf (with the complementary probability to the 0-node). The way the network externally samples is from the highest probability for each digit, ultimately saving space but destroying the conditional probability for further digits in the sequence. Here's one such case: there could be some case where the total sum of the magnitude of the likelihoods of one branch high in the network is higher than a (largely) single-valued magnitude of likelihoods from the opposing branch, and at some point during the binary tree traversal if those magnitudes have separate, distinct path traversals as the tree is navigated down the branches of the most likely weights, then the subtree of the single large distribution can overwhelm the subtree selection of the multiple smaller distribution magnitudes and point to a nonexistant distribution.

   Here's an example: Suppose that at the top node, the right branch of the tree (represented by 1 in this case) contains a single, high magnitude cluster of values, and suppose that the left branch (in this case represented by 0) contains several lower-magnitude clusters whose sum is larger than the right hand cluster. Thus one failure case (confidence values are in the respective [0, 1] positions), we can represent such a distribution with a confidence value of [.56, .44] for the first digit of the bin distribution. Assuming before softmax the values of the lower left hand branches of the tree have a high magnitude but similar values, we can suppose a root node where the left hand pre-softmax values have weight contributions [7, 3] as compared to the right hand branch's weight contribution of \[1, 4\] (In reality, this information is entangled together, but for the sake of example we are separating out the values). Suppose one step down the magnitude of the left hand contributions' split is [4,4] and the right hand contributions' split is [1,3]. After softmax, the right hand sub-branch will likely be chosen from that point on, even though the digit for the left hand branch was chosen at first. To recap, to this point in the example, the digit first picked was 0 and then 1, because the confidence sampling from digit to digit is internally contained in the network, instead of sampling with external, explicit sampling conditions. So moving forward, if the right branch has distributions which pick a digit choice of 0, 1, 1, 0 for the remaining digits, then traversing that path on the left hand branch of the binary tree may will be "sampled" as the bin location with the highest probability magnitude, but may not actually contain a probability value of that magnitude, opening the door to incorrect sampling from a distribution that properly satisfies the cross-entropy loss.

   It is still unclear to me how much of a negative effect this has in practice, as I would suspect this would come more sharply into play the more generalized a network is to a task. It is however, one of the side effects of independently choosing binary digits from combined distribution magnitudes, and a natural consequence of that lossy compression.

   One solution to this problem is individually storing the weights of each branch and doing an externally conditional traversal. However, since the number of branch weights would equal the number of desired bin values, from both a memory and convergence standpoint I don't see any immediate benefits from this alternative over the normal classification output, which in the case of colorization seems to be better suited due to the similarity of co-located bins.

As it stands, it's a network I'm proud of. I personally believe that it has a lot of future potential.

If you have any questions about this network, the first network, or any of my other projects, feel free to drop me a line at my personal email, tbcharger@gmail.com.

## 3. Floating Point Division:

   This was for a class project in the fall of 2015, and is a more vanilla demonstration of my coding ability. It is a relatively simple, lower level program to do floating point division on an input file of two floating point integers in hex, each pair separated by a space, one per line. Compared to the networks above, it may be more boring, but I think it's a good demonstration of my (somewhat) lower level abilities since the above networks were mostly implemented using high-level APIs.

## 4. Pix2Pix/StackGAN Adapted for Colorization 

   I'm currently working on the problem of preserving the spatial information of an input image while still maintaining a StackGAN-like structure. The more I work with the network (sourced from a TensorFlow adaptation of Pix2Pix), the less it looks like Pix2Pix and more like StackGAN. I've initially disabled the U-Net (or bridged) layers from the encoder to the decoder to force information flow exclusively through that high dimensional bottleneck, which will later allow for the conditional augmentation and KL loss that StackGAN uses. The implementation I found (and the paper appears to confirm this) doesn't seem to have much in the way of noise (which makes sense since nearly allof the  encoder-decoder layers are directly bridged, and additive noise could easily be ignored by the network. I'm not sure about concatenated noise, though). I added a z vector of noise drawn from a normal distribution concatenated to the normal high-level vector at that point. One issue I'm floating around is preserving spatial information. Training a secondary conditional augmentation sub-network in the same way StackGAN does for the second part of the network may work, though it may more effective to have a fully separately trainable convolutional stack before the conditional augmentation for the input image. This is because the vanilla StackGAN uses draws from the conditioned text embeddings from a pre-trained network, which as far as I know do not have a large amount of spatial information compared to input images.

   One trick that will likely come back to save the day for Colorization is Chroma Subsampling. I'm currently working with the facades dataset as a small-scale dataset, (generation instead of colorization) since I'd prefer to create a general case solution and then adapt it for a specific use rather than the vice versa.

   For the future of this project, this is probably the way I will take the colorization network. I like the concept of using GANs for colorization (or information adding) since there's no need for a high-depth external representation, unlike the classification network. This also seems to have rapid convergence with fewer parameters when compared to the classification network I used.
