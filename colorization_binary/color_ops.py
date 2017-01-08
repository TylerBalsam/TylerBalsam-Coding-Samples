import tensorflow as tf
import numpy as np

x_test = np.float32(np.arange(15,0.1))
NTEST = x_test.size
x_test = x_test.reshape(NTEST,1) # needs to be a matrix, not a vector

def round_to_4_times(val, base_val = 1):
	if val < base_val:
		return base_val
	else:
		return round_to_4_times(val, 4*base_val)

def _phase_shift(I, r):
    # Helper function with main phase shift operation
    bsize, a, b, c = I.get_shape().as_list()
    X = tf.reshape(I, (bsize, a, b, r, r))
    X = tf.transpose(X, (0, 1, 2, 4, 3))  # bsize, a, b, 1, 1
    X = tf.split(1, a, X)  # a, [bsize, b, r, r]
    X = tf.concat(2, [tf.squeeze(x) for x in X])  # bsize, b, a*r, r
    X = tf.split(1, b, X)  # b, [bsize, a*r, r]
    X = tf.concat(2, [tf.squeeze(x) for x in X])  #
    bsize, a*r, b*r
    return tf.reshape(X, (bsize, a*r, b*r, 1))

def PS(X, r, color=False):
  # Main OP that you can arbitrarily use in you tensorflow code
  if color:
    Xc = tf.split(3, X.get_shape()[3]/(r*r), X)
    X = tf.concat(3, [_phase_shift(x, r) for x in Xc])
  else:
    X = _phase_shift(X, r)
  return X

def PhaseShift_BatchToSpace(X, r):
 X = tf.transpose(X, (3, 1, 2, 0))
 result = tf.batch_to_space(X, [[0, 0], [0, 0]], r)
 result = tf.transpose(result, (3, 1, 2, 0))
 return result

def get_pi_idx(x, pdf):
  N = pdf.size
  accumulate = 0
  for i in range(0, N):
    accumulate += pdf[i]
    if (accumulate >= x):
      return i
  print 'error with sampling ensemble'
  return -1

def generate_ensemble(out_pi, out_mu, out_sigma, N, M = 10):
  NTEST = x_test.size
  result = np.random.rand(NTEST, M) # initially random [0, 1]
  rn = np.random.randn(NTEST, M) # normal random matrix (0.0, 1.0)
  out_pi1 = out_pi[-1, -1, 0]#np.reshape(out_pi, 50176)
  out_mu1 = out_mu[-1, -1, 0]#np.reshape(out_mu, 50176)
  out_sigma1 = out_sigma[-1, -1, 0]
  out_pi2 = out_pi[-1, -1, 3]#np.reshape(out_pi, 50176)
  out_mu2 = out_mu[-1, -1, 3]#np.reshape(out_mu, 50176)
  out_sigma2 = out_sigma[-1, -1, 3]

  mu = 0
  std = 0
  idx = 0
  
  # transforms result into random ensembles
  for j in range(0, 224):
    for i in range(0, 224):
      idx1 = get_pi_idx(result[i, j], out_pi1[i])
      mu1 = out_mu1[i, idx]
      std1 = out_sigma1[i, idx]
      result[i, j, 0] = mu1 + rn[i, j]*std1
      idx2 = get_pi_idx(result[i, j], out_pi2[i])
      mu2 = out_mu2[i, idx]
      std2 = out_sigma2[i, idx]
      result[i, j, 1] = mu2 + rn[i, j]*std1
  return result


def rgb2yuv(rgb):
    """
    Convert RGB image into YUV https://en.wikipedia.org/wiki/YUV
    """
    rgb2yuv_filter = tf.constant(
        [[[[0.299, -0.169, 0.499],
           [0.587, -0.331, -0.418],
            [0.114, 0.499, -0.0813]]]])
    rgb2yuv_bias = tf.constant([0., 0.5, 0.5])

    temp = tf.nn.conv2d(rgb, rgb2yuv_filter, [1, 1, 1, 1], 'SAME')
    temp = tf.nn.bias_add(temp, rgb2yuv_bias)

    return temp


def yuv2rgb(yuv):
    """
    Convert YUV image into RGB https://en.wikipedia.org/wiki/YUV
    """
    yuv = tf.mul(yuv, 255)
    yuv2rgb_filter = tf.constant(
        [[[[1., 1., 1.],
           [0., -0.34413999, 1.77199996],
            [1.40199995, -0.71414, 0.]]]])
    yuv2rgb_bias = tf.constant([-179.45599365, 135.45983887, -226.81599426])
    temp = tf.nn.conv2d(yuv, yuv2rgb_filter, [1, 1, 1, 1], 'SAME')
    temp = tf.nn.bias_add(temp, yuv2rgb_bias)
    temp = tf.maximum(temp, tf.zeros(temp.get_shape(), dtype=tf.float32))
    temp = tf.minimum(temp, tf.mul(
        tf.ones(temp.get_shape(), dtype=tf.float32), 255))
    temp = tf.div(temp, 255)
    return temp

def yuv2rgbnp(im):
  """convert array-like yuv image to rgb colourspace

  a pure numpy implementation since the YCbCr mode in PIL is b0rked.  
  """
  ## conflicting definitions exist depending on whether you use the full range
  ## of YCbCr or clamp out to the valid range.  see here
  ## http://www.equasys.de/colorconversion.html
  ## http://www.fourcc.org/fccyvrgb.php
  from numpy import dot, ndarray, array
  if not im.dtype == 'uint8':
    raise ImageUtilsError('yuv2rgb only implemented for uint8 arrays')

  ## better clip input to the valid range just to be on the safe side
  yuv = ndarray(im.shape)  ## float64
  yuv[:,:, 0] = im[:,:, 0].clip(16, 235).astype(yuv.dtype) - 16
  yuv[:,:,1:] = im[:,:,1:].clip(16, 240).astype(yuv.dtype) - 128

  ## ITU-R BT.601 version (SDTV)
  A = array([[1.,                 0.,  0.701            ],
             [1., -0.886*0.114/0.587, -0.701*0.299/0.587],
             [1.,  0.886,                             0.]])
  A[:,0]  *= 255./219.
  A[:,1:] *= 255./112.

  ## ITU-R BT.709 version (HDTV)
#  A = array([[1.164,     0.,  1.793],
#             [1.164, -0.213, -0.533],
#             [1.164,  2.112,     0.]])

  rgb = dot(yuv, A.T)
  return rgb.clip(0, 255).astype('uint8')
