import cv2
from scipy.ndimage import median_filter

im = cv2.imread('mask_new.jpg').mean(axis=2)
im[(im > 1)] = 255.
im = median_filter(im, size=(3, 3))
cv2.imwrite('test_mask.jpg', im)