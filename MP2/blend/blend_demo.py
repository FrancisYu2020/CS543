# Code from Saurabh Gupta
import cv2
from scipy.ndimage import gaussian_filter
import matplotlib.pyplot as plt
import numpy as np
from absl import app, flags

from blend_solve import blend

FLAGS = flags.FLAGS
flags.DEFINE_string('im1', 'apple.jpeg', 'image 1')
flags.DEFINE_string('im2', 'orange.jpeg', 'image 2')
flags.DEFINE_string('mask', 'mask.png', 'mask image')
flags.DEFINE_string('out_name', 'output.png', 'output image name')

def main(_):
  I1 = cv2.imread(FLAGS.im1)
  I2 = cv2.imread(FLAGS.im2)
  mask = cv2.imread(FLAGS.mask)
#   black = np.zeros(I1.shape)
  
  out = blend(I1, I2, mask)
  # save image
  cv2.imwrite(FLAGS.out_name, out)
#   cv2.imwrite('1' + FLAGS.out_name, blend(I1, mask, mask))
#   cv2.imwrite('2' + FLAGS.out_name, blend(mask, I2, mask))

if __name__ == '__main__':
  app.run(main)
