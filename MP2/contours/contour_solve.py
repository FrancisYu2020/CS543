import numpy as np
from scipy import signal
import cv2
from copy import copy
from scipy.ndimage import gaussian_filter

def smoothing(input, sigma, dir):
    assert dir == 0 or dir == 1, "dir = 0 for dx and dir = 1 for dy"
    # For bells and whistles, uncomment to run this part
    # a = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1]]) #For bells and whistle 1
    # a = np.array([[1, 1, 1, 1, 0, 1, 1, 1, 1]]) #For bells and whistle 2
    # filters = [np.dot(a.T, a), np.array([[-1, 0, 1.]])]
    # if dir:
    #     filters = [filter.T for filter in filters]
    # for filter in filters:
    #     input = signal.convolve2d(input, filter, mode='same', boundary='symm')
    # return input
    input = gaussian_filter(input, sigma)
    derivative = np.array([[-1, 0, 1]])
    if dir:
        derivative = derivative.T
    input = signal.convolve2d(input, derivative, mode='same', boundary='symm')
    return input

def bells_and_whistles(input, dir):
    # For bells and whistles, uncomment to run this part
    # a = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1]]) #For bells and whistle 1
    a = np.array([[1, 1, 1, 1, 0, 1, 1, 1, 1]]) #For bells and whistle 2
    filters = [np.dot(a.T, a), np.array([[-1, 0, 1.]])]
    if dir:
        filters = [filter.T for filter in filters]
    for filter in filters:
        input = signal.convolve2d(input, filter, mode='same', boundary='symm')
    return input

def NMS(img, theta):
    Z = img - img
    M, N = Z.shape
    theta = theta * 180. / np.pi
    theta[theta < 0] += 180
    for i in range(1,M-1):
        for j in range(1,N-1):
            q = 255
            r = 255

            if (0 <= theta[i,j] < 22.5) or (157.5 <= theta[i,j] <= 180):
                q = img[i, j+1]
                r = img[i, j-1]

            elif (22.5 <= theta[i,j] < 67.5):
                q = img[i+1, j-1]
                r = img[i-1, j+1]

            elif (67.5 <= theta[i,j] < 112.5):
                q = img[i+1, j]
                r = img[i-1, j]

            elif (112.5 <= theta[i,j] < 157.5):
                q = img[i-1, j-1]
                r = img[i+1, j+1]

            if (img[i,j] >= q) and (img[i,j] >= r):
                Z[i,j] = img[i,j]
            else:
                Z[i,j] = 0
    return Z/Z.max()*255

def compute_edges_dxdy(I):
    """Returns the norm of dx and dy as the edge response function."""
    I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
    I = I.astype(np.float32)/255.

    #Initial implement
    # dx = signal.convolve2d(I, np.array([[-1, 0, 1]]), mode='same')
    # dy = signal.convolve2d(I, np.array([[-1, 0, 1]]).T, mode='same')

    #Part 1: uncomment to run part 1
    # dx = signal.convolve2d(I, np.array([[-1, 0, 1]]), mode='same', boundary='symm')
    # dy = signal.convolve2d(I, np.array([[-1, 0, 1]]).T, mode='same', boundary='symm')

    #Part 2 smoothing:
    # dx = smoothing(I, 2, 0)
    # dy = smoothing(I, 2, 1)

    #Bells and whistles 2:
    dx = bells_and_whistles(I, 0)
    dy = bells_and_whistles(I, 1)

    #Common part before NMS
    mag = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)

    #Part 3 NMS:
    mag = NMS(mag, theta)

    #Common part, no need to modify
    mag = mag / mag.max() * 255
    mag = np.clip(mag, 0, 255)
    mag = mag.astype(np.uint8)
    return mag
