from __future__ import print_function
import sys
import re
from skimage import io, color
from PIL import Image, ImageOps
from scipy import ndimage, misc
import scipy
import numpy as np
import h5py
import os

np.set_printoptions(threshold=np.nan)
images = []
filename_list = []
color_images = []
bw_fin_images = []
err_fin_images = []
total_pixels = 100352
bad_imgs = []
size = 224, 224
chunk_size = 250
partial_chunks = False

root_dir = "/home/tyler/Documents/"

if "partial_chunks" in sys.argv[1:]:
	partial_chunks = True

if "train" in sys.argv[1:]:
	source_dir= "MSCOCO/train2014"
	dset_name= "train_images.hdf5"
if "test" in sys.argv[1:]:
	source_dir= "MSCOCO/test"
	dset_name = "test_images.hdf5"

def _ycc(r, g, b): # in (0,255) range
    y = .299*r + .587*g + .114*b
    cb = 128 -.168736*r -.331364*g + .5*b
    cr = 128 +.5*r - .418688*g - .081312*b
    return y, cb, cr

#NOTE: "!" preceding a comment means high importance -- "do firsts" if you will.
#!TODO: Investigate 400 chunk cap.
#TODO: Make COLOR_IMAGES... an opt to pass in.
#TODO: Use absolute paths as specified above...
#This attempts to remove the dataset if it already exists.

try:
    os.remove(dset_name)
except OSError:
    pass

for root, dirnames, filenames in os.walk(root_dir+source_dir):
	filename_list = filenames

chunk_num = len(filename_list)/chunk_size

print("Creating "+str(chunk_num)+" chunks for the dataset")

with h5py.File(dset_name, 'w') as hf:
	color_dset = hf.create_dataset('ycbcr_imgs', (10, 224, 224, 3), maxshape=(None, 224, 224, 3))

	for i in range(len(filename_list)):
	        filepath = os.path.join(root_dir+source_dir+'/', filename_list[i])
	        image = ndimage.imread(filepath, mode="RGB")
		image = Image.fromarray(image)
	        image = ImageOps.fit(image, size, Image.ANTIALIAS)
		image = np.reshape(np.array(image), [224, 224, 3])
		list = np.dsplit(image, 3)
		image = np.dstack(_ycc(list[0], list[1], list[2]))
		image = image/255
	        color_images.append(image)
	        # First condition allows for partial chunks if specified as a param."
		if ((partial_chunks) and ((i + 1) == len(filename_list))) or ((i + 1) % chunk_size == 0):
		    if (i + 1) == len(filename_list): #If this is a partial chunk, set the offset to that length
			backwards_offset = len(filename_list) % chunk_size
		    else:
			backwards_offset = chunk_size
		    print("Chunk "+str((i+1)/chunk_size)+" of "+str(chunk_num)+".")
		    # Careful for off-by-one errors here.
		    color_dset.resize(i+1, axis=0)
		    color_images = np.array(color_images)
		    color_dset[(i+1)-backwards_offset:, :, :, :] = color_images
		    color_images = []
