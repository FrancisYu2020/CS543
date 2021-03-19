import numpy as np
import scipy
from scipy import signal

def convseq(input):
    input = scipy.ndimage.gaussian_filter(input, 0.55, order=0, output=None, mode='reflect')
    return input

def derivative(input, dir):
    assert dir == 0 or dir == 1, "dir = 0 for dx and dir = 1 for dy"
    filter = np.array([[-1, 0, 1.]])
    if dir:
        filter = filter.T
    input = signal.convolve2d(input, filter, mode='same', boundary='symm')
    return input

def compute_corners(I):
  # Currently this code proudces a dummy corners and a dummy corner response
  # map, just to illustrate how the code works. Your task will be to fill this
  # in with code that actually implements the Harris corner detector. You
  # should return th ecorner response map, and the non-max suppressed corners.
  # Input:
  #   I: input image, H x W x 3 BGR image
  # Output:
  #   response: H x W response map in uint8 format
  #   corners: H x W map in uint8 format _after_ non-max suppression. Each
  #   pixel stores the score for being a corner. Non-max suppressed pixels
  #   should have a low / zero-score.

  I = np.mean(I, axis=2)
  dx = derivative(I, 0)
  dy = derivative(I, 1)
  Mx = dx**2
  My = dy**2
  Mxy = dx*dy
  Mx = convseq(Mx)
  My = convseq(My)
  Mxy = convseq(Mxy)
  H, W = I.shape
  alpha = 0.04
  response = Mx*My - Mxy**2 - alpha*(Mx + My)**2

  corners = response / response.max() * 255
  window_size = 3
  for h in range(0, H):
      for w in range(0, W):
          for i in range(max(0, h - window_size), min(h + window_size + 1, H)):
              for j in range(max(0, w - window_size), min(w + window_size + 1, W)):
                  if response[i, j] > response[h, w]:
                      corners[h, w] = 0

  corners = corners.astype(np.uint8)

  response = response * 255.
  response = np.clip(response, 0, 255)
  response = response.astype(np.uint8)

  return response, corners
