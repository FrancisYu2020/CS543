import numpy as np
from scipy.ndimage import gaussian_filter
from skimage.filters import difference_of_gaussians, window, gaussian
import matplotlib.pyplot as plt
import cv2

# def blend(im1, im2, mask): #TODO not quite works
    #     init_sigma = 1
    #     # lap stack of im1
    #     im1_lap0 = im1 - gaussian_filter(im1, init_sigma, order=0, output=None, mode='reflect')
    #     im1_lap1 = gaussian_filter(im1, init_sigma, order=0, output=None, mode='reflect') - gaussian_filter(im1, init_sigma*2, order=0, output=None, mode='reflect') 
    #     im1_lap2 = gaussian_filter(im1, init_sigma*2, order=0, output=None, mode='reflect') - gaussian_filter(im1, init_sigma*4, order=0, output=None, mode='reflect') 
    #     im1_lap3 = gaussian_filter(im1, init_sigma*4, order=0, output=None, mode='reflect') - gaussian_filter(im1, init_sigma*8, order=0, output=None, mode='reflect') 
    #     im1_lap4 = gaussian_filter(im1, init_sigma*8, order=0, output=None, mode='reflect') 

    #     # lap stack of im2
    #     im2_lap0 = im2 - gaussian_filter(im2, init_sigma, order=0, output=None, mode='reflect')
    #     im2_lap1 = gaussian_filter(im2, init_sigma, order=0, output=None, mode='reflect') - gaussian_filter(im2, init_sigma*2, order=0, output=None, mode='reflect') 
    #     im2_lap2 = gaussian_filter(im2, init_sigma*2, order=0, output=None, mode='reflect') - gaussian_filter(im2, init_sigma*4, order=0, output=None, mode='reflect') 
    #     im2_lap3 = gaussian_filter(im2, init_sigma*4, order=0, output=None, mode='reflect') - gaussian_filter(im2, init_sigma*8, order=0, output=None, mode='reflect') 
    #     im2_lap4 = gaussian_filter(im2, init_sigma*8, order=0, output=None, mode='reflect') 

    #     # gussian stack of mask
    #     mask_0 = mask
    #     mask_1 = gaussian_filter(mask, init_sigma, order=0, output=None, mode='reflect')
    #     mask_2 = gaussian_filter(mask, init_sigma*2, order=0, output=None, mode='reflect')
    #     mask_3 = gaussian_filter(mask, init_sigma*4, order=0, output=None, mode='reflect')
    #     mask_4 = gaussian_filter(mask, init_sigma*8, order=0, output=None, mode='reflect')


    #     mask_0 = mask_0 / 255.
    #     mask_1 = mask_1 / 255.
    #     mask_2 = mask_2 / 255.
    #     mask_3 = mask_3 / 255.
    #     mask_4 = mask_4 / 255.

    #     out_0 = im1_lap0 * mask_0 + (1-mask_0) * im2_lap0
    #     out_1 = im1_lap1*mask_1 + (1-mask_1)*im2_lap1
    #     out_2 = im1_lap2*mask_2 + (1-mask_2)*im2_lap2
    #     out_3 = im1_lap3*mask_3 + (1-mask_3)*im2_lap3
    #     out_4 = im1_lap4 * mask_4 + (1-mask_4) * im2_lap4

    #     plt.imshow(out_0.astype('uint8'))
    #     plt.show()
    #     plt.imshow(out_1.astype('uint8'))
    #     plt.show()
    #     plt.imshow(out_2.astype('uint8'))
    #     plt.show()
    #     plt.imshow(out_3.astype('uint8'))
    #     plt.show()
    #     plt.imshow(out_4.astype('uint8'))
    #     plt.show()
    #     # out = out_0
    #     # out = out_0/5. + out_1/5. + out_2/5. + out_3/5. + out_4/5.
    #     # out = out_0 + out_1/2. + out_2/4. + out_3/8. + out_4/16.
    #     out = out_0 + out_1 + out_2 + out_3 + out_4
    #     # out = out.astype('uint8')
    #     plt.imshow(out.astype('uint8'))
    #     plt.show()
        
    #     # out = im1 * mask + (1-mask) * im2
    #     return out
def resize(im1,im2,mask):
    # resize the image to be same size as mask
    resized_im1=cv2.resize(im1, mask.shape[:2])
    resized_im2=cv2.resize(im2, mask.shape[:2])
    return resized_im1, resized_im2

def generate_mask(size, mode = 'vertical', flip = False):
    canvas = np.zeros((size,size,3),dtype=np.uint8)
    if mode == 'vertical':
        if flip:
            canvas[:,int(size/2):,:] = 255
        else:
            canvas[:,:int(size/2),:] = 255
    elif mode == 'horizontal':
        if not flip:
            canvas[:int(size/2),:,:] = 255
        else:
            canvas[int(size/2):,:,:] = 255
    else:
        raise NotImplementedError
    return canvas

def show_im(img):
    plt.imshow(img)
    plt.show()

def normalize_img(img):
    return (((img - np.min(img)) / (np.max(img) - np.min(img)))*255.0).astype('uint8')

def blend(im1, im2, mask):

    init_sigma = 10
    # difference of gaussian stack
    
    # im1
    lap_im1_1 = difference_of_gaussians(im1,0,init_sigma,mode='reflect')
    lap_im1_2 = difference_of_gaussians(im1,init_sigma,init_sigma*2,mode='reflect')
    lap_im1_3 = difference_of_gaussians(im1,init_sigma*2,init_sigma*4,mode='reflect')
    lap_im1_4 = difference_of_gaussians(im1,init_sigma*4,init_sigma*8,mode='reflect')
    lap_im1_5 = difference_of_gaussians(im1,init_sigma*8,init_sigma*16,mode='reflect')
    lap_im1_6 = difference_of_gaussians(im1,init_sigma*16,init_sigma*32,mode='reflect')
    
    
    # im2
    lap_im2_1 = difference_of_gaussians(im2,0,init_sigma,mode='reflect')
    lap_im2_2 = difference_of_gaussians(im2,init_sigma,init_sigma*2,mode='reflect')
    lap_im2_3 = difference_of_gaussians(im2,init_sigma*2,init_sigma*4,mode='reflect')
    lap_im2_4 = difference_of_gaussians(im2,init_sigma*4,init_sigma*8,mode='reflect')
    lap_im2_5 = difference_of_gaussians(im2,init_sigma*8,init_sigma*16,mode='reflect')
    lap_im2_6 = difference_of_gaussians(im2,init_sigma*16,init_sigma*32,mode='reflect')
    
    # mask
    mask_1 = gaussian(mask,init_sigma,multichannel=True)
    mask_2 = gaussian(mask,init_sigma*2,multichannel=True)
    mask_3 = gaussian(mask,init_sigma*4,multichannel=True)
    mask_4 = gaussian(mask,init_sigma*8,multichannel=True)
    mask_5 = gaussian(mask,init_sigma*16, multichannel=True)
    mask_6 = gaussian(mask,init_sigma*32, multichannel=True)

    # out
    out_1 = lap_im1_1*mask_1 + (1-mask_1)*lap_im2_1
    out_2 = lap_im1_2*mask_2 + (1-mask_2)*lap_im2_2
    out_3 = lap_im1_3*mask_3 + (1-mask_3)*lap_im2_3
    out_4 = lap_im1_4*mask_4 + (1-mask_4)*lap_im2_4
    out_5 = lap_im1_5*mask_5 + (1-mask_5)*lap_im2_5
    out_6 = lap_im1_6*mask_6 + (1-mask_6)*lap_im2_6

    out = out_1 + out_2 + out_3 + out_4 + out_5 + out_6
    out = normalize_img(out)

    # show_im(out)
    
        # im1_blur_1 = gaussian_filter(im1, init_sigma, order=0, mode='reflect')
        # im1_blur_2 = gaussian_filter(im1, init_sigma*2, order=0, mode='reflect')
        # def_1 = im1 - im1_blur_1
        # def_2 = im1_blur_1 - im1_blur_2

        # def_2_ = difference_of_gaussians(im1,0,1,mode='reflect')
        # def_2_ = ((def_2_ - np.min(def_2_)) / (np.max(def_2_) - np.min(def_2_)))*255.0
        # cv2.imwrite('test_DoG.png',def_2_)
        # show_im(im1)
        # show_im(im1_blur_1)
        # show_im(im1_blur_2)
        # show_im(def_2)
        # show_im(def_2_.astype('uint8'))

    return out

