import numpy as np
from scipy import signal
import cv2
from copy import copy

def convseq(input, dir):
    assert dir == 0 or dir == 1, "dir = 0 for dx and dir = 1 for dy"
    a = np.array([[1, 8, 28, 56, 70, 56, 28, 8, 1]])
    # a = np.array([[1, 1, 1, 1, 0, 1, 1, 1, 1]])
    # sigma1 = np.array([[1, 4, 7, 4, 1],
    #                    [4, 16, 26, 16, 4],
    #                    [7, 26, 41, 26, 7],
    #                    [4, 16, 26, 16, 4],
    #                    [1, 4, 7, 4, 1]])
    filters = [np.dot(a.T, a), np.array([[-1, 0, 1.]])]
    if dir:
        filters = [filter.T for filter in filters]
    for filter in filters:
        input = signal.convolve2d(input, filter, mode='same', boundary='symm')
    return input

def non_max_suppression(img, D):
    M, N = img.shape
    Z = np.zeros((M,N))
    angle = D * 180. / np.pi
    angle[angle < 0] += 180
    for i in range(1,M-1):
        for j in range(1,N-1):
            q = 255
            r = 255
           #angle 0
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                q = img[i, j+1]
                r = img[i, j-1]
            #angle 45
            elif (22.5 <= angle[i,j] < 67.5):
                q = img[i+1, j-1]
                r = img[i-1, j+1]
            #angle 90
            elif (67.5 <= angle[i,j] < 112.5):
                q = img[i+1, j]
                r = img[i-1, j]
            #angle 135
            elif (112.5 <= angle[i,j] < 157.5):
                q = img[i-1, j-1]
                r = img[i+1, j+1]

            if (img[i,j] >= q) and (img[i,j] >= r):
                Z[i,j] = img[i,j]
            else:
                Z[i,j] = 0
    return Z/Z.max()*255

# def threshold(img, lowThresholdRatio=0.3, highThresholdRatio=0.95):
#
#     highThreshold = img.max() * highThresholdRatio;
#     lowThreshold = highThreshold * lowThresholdRatio;
#
#     M, N = img.shape
#     res = np.zeros((M,N))
#
#     weak = np.int32(55)
#     strong = np.int32(255)
#
#     strong_i, strong_j = np.where(img >= highThreshold)
#     zeros_i, zeros_j = np.where(img < lowThreshold)
#
#     weak_i, weak_j = np.where((img <= highThreshold) & (img >= lowThreshold))
#
#     res[strong_i, strong_j] = strong
#     res[weak_i, weak_j] = weak
#
#     return (res, weak, strong)
#
#
# def hysteresis(img, weak=25, strong=255):
#     M, N = img.shape
#     for i in range(1, M-1):
#         for j in range(1, N-1):
#             if (img[i,j] == weak):
#                 try:
#                     if ((img[i+1, j-1] == strong) or (img[i+1, j] == strong) or (img[i+1, j+1] == strong)
#                         or (img[i, j-1] == strong) or (img[i, j+1] == strong)
#                         or (img[i-1, j-1] == strong) or (img[i-1, j] == strong) or (img[i-1, j+1] == strong)):
#                         img[i, j] = strong
#                     else:
#                         img[i, j] = 0
#                 except IndexError as e:
#                     pass
#     return img

def compute_edges_dxdy(I):
  """Returns the norm of dx and dy as the edge response function."""
  I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
  I = I.astype(np.float32)/255.
  dx = convseq(I, 0)
  dy = convseq(I, 1)
  mag = np.sqrt(dx**2 + dy**2)
  theta = np.arctan2(dy, dx)
  mag = non_max_suppression(mag, theta)
  # mag, weak, strong = threshold(mag)
  # mag = hysteresis(mag, weak, strong)
  mag = mag / mag.max()
  mag = mag * 255.
  mag = np.clip(mag, 0, 255)
  mag = mag.astype(np.uint8)
  return mag
