import h5py
import sys
from color_ops import *
from skimage import io, color
import scipy
import numpy as np
import time
import tflearn
from tflearn.layers.normalization import batch_normalization
from tflearn.layers.core import *
from tflearn.layers.conv import *
from tflearn.layers.estimator import regression
# TODO: Add getopt for args

# The input data contains the full color image data in the ycbcr
# format. It is split into BW and color data, and fitted against
# the pre-split input data. 
data = h5py.File('train_images.hdf5', 'r')
X = data['ycbcr_imgs']

test_data = h5py.File('test_images.hdf5', 'r')
Z = test_data['ycbcr_imgs']

# Hyperparams (Note: BATCH_NUM of 1 can cause issues because of
# tf.squeeze use throughout. Small BATCH_NUMs are generally not
# recommended, however.)
BATCH_NUM = 20
LEARNING_RATE = .001
EPOCH_NUM = 2
FILTERS = 8

# Forward pass without training. Should be equal to test dataset
# size.
if not ("train" in sys.argv[1:]):
	BATCH_NUM = 80

# Network layer reference vars
BRIDGE_STACK = []

# Network Architecture Params
BIN_SIZE = 25
RECURSION_LENGTH = 4
RECURSION_NUM = 2
LAYER_DEPTH = 5
LAYER_BLOCK_NUM = 3
CENTER_BLOCK_NUM = 2

tflearn.init_graph(num_cores=0, gpu_memory_fraction=0)

# Network Input
network = input_data(shape=[None, 224, 224, 3])
network = tf.reshape(network, [BATCH_NUM, 224, 224, 3])

# Split inputs to BW and color
network = tf.unpack(network, axis = 3)
bw = tf.expand_dims(network.pop(0), 3)
col = tf.pack(network, 3)
network = tf.reshape(bw, [BATCH_NUM, 224, 224, 1])

# Encoder
for i in range(LAYER_BLOCK_NUM):
    for f in range(LAYER_DEPTH-1):
        network = conv_2d(network, FILTERS, 3, activation='elu')
    network = batch_normalization(network)
    BRIDGE_STACK.append(network)
    FILTERS = FILTERS*2
    network = conv_2d(network, FILTERS, 3, 2, activation='elu')

# Bottleneck
for i in range(CENTER_BLOCK_NUM):
    network = conv_2d(network, FILTERS, 3, activation='elu')
    network = batch_normalization(network)

# Decoder to second bottleneck
for i in range(LAYER_BLOCK_NUM):
    # Divides depth by 4 by expanding 1 x 1 x 4 blocks into
    # 2 x 2 x 1 blocks. The equivalent layer from the encoder
    # network is concatenated as well. PhaseShift inspired by
    # https://github.com/Tetrachrome/subpixel.
    network = PhaseShift_BatchToSpace(network, 2)
    tf.concat(3, [network, BRIDGE_STACK.pop()])

    for j in range(LAYER_DEPTH-1):
        network = highway_conv_2d(network, FILTERS, 3, activation='elu')
    network = batch_normalization(network)
    FILTERS = FILTERS/4

# Quarter depth classification
network = conv_2d(network, BIN_SIZE/2, 3, activation = 'elu')
network = batch_normalization(network)

# Half depth classification
network = conv_2d(network, BIN_SIZE, 1, activation ='elu')
network = batch_normalization(network)

# Full depth classification (Large)
network = conv_2d(network, BIN_SIZE*2, 1, activation ='elu')
network = batch_normalization(network)

if "train" in sys.argv[1:]:
	network = tf.nn.softmax(tf.reshape(network, [-1, BIN_SIZE]))
	network = tf.pow(network, 2)
	network = tf.reshape(network, [BATCH_NUM, 224, 224, 2, BIN_SIZE])

	# Boolean mask to select confidence values at the correct
	# bin location. With the frequent network changes I've made
	# experimenting, passing these confidence values as if they
	# were discrete values and having them unpacked by a custom
	# loss function is more flexible. For example, it opened up
	# the ability to experiment with a pow(confidence, -4) * true
	# value input to a VGG16 content loss.
        col = col * tf.to_float(BIN_SIZE-1)
        one_hot = tf.one_hot(tf.to_int32(col), BIN_SIZE, on_value=True, off_value=False, dtype=tf.bool)
	network = tf.boolean_mask(network, one_hot)
	network = tf.reshape(network, [BATCH_NUM, 224, 224, 2])
	network = tf.concat(3, [bw, network])
else:
	network = tf.reshape(network, [BATCH_NUM, 224, 224, 2, BIN_SIZE])
	max = tf.squeeze(tf.argmax(network, 4))
	network = tf.to_float(max) * tf.inv(tf.to_float(BIN_SIZE)-1)
	network = tf.concat(3, [bw, network])
	network = yuv2rgb(network)

# Regression & loss designation.
if "train" in sys.argv[1:]:
	network = regression(network, optimizer='Adam',
	                     loss='neg_ln_with_total_variance',
	                     learning_rate=LEARNING_RATE,
			     metric=None)

model = tflearn.DNN(network, checkpoint_path='model',
                    max_checkpoints=1, tensorboard_verbose=1)

if "load" in sys.argv[1:] or not ("train" in sys.argv[1:]):
	model.load("model.tflearn")

if "train" in sys.argv[1:]:
	model.fit(X, X, n_epoch=EPOCH_NUM, shuffle=True,
	          show_metric=False, batch_size=BATCH_NUM, snapshot_step=3000,
	          snapshot_epoch=False, run_id='colorize')

	model.save("model.tflearn")

# At the end of training, or in normal run, predict
# and save the resulting colorized images.
out = np.array(model.predict(Z))
out = out*255
out = out.astype(np.uint8)
out = np.split(out, out.shape[0])
i = 0
for val in out:
	scipy.misc.toimage(np.squeeze(val), cmin=0, cmax=255).save('results/colorized'+str(i)+'.png')
	i += 1
