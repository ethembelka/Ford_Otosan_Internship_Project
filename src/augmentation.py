

# importing all the required libraries
import warnings
warnings.filterwarnings('ignore')
import numpy as np
import skimage.io as io
from skimage.transform import rotate, AffineTransform, warp
from skimage.util import random_noise
from skimage.filters import gaussian
import matplotlib.pyplot as plt
import cv2
from constant import *
import os
import tqdm
from torchvision import transforms as T
from PIL import Image


image_list = os.listdir(IMAGE_DIR)
mask_list = os.listdir(MASK_DIR)

def augment_shadow(img):
    top_y = 320 * np.random.uniform()
    top_x = 0
    bot_x = 160
    bot_y = 320 * np.random.uniform()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    shadow_mask = 0 * hsv[:, :, 1]
    X_m = np.mgrid[0:img.shape[0], 0:img.shape[1]][0]
    Y_m = np.mgrid[0:img.shape[0], 0:img.shape[1]][1]

    shadow_mask[((X_m - top_x) * (bot_y - top_y) - (bot_x - top_x) * (Y_m - top_y) >= 0)] = 1

    shadow_density = .5
    left_side = shadow_mask == 1
    right_side = shadow_mask == 0

    if np.random.randint(2) == 1:
        hsv[:, :, 2][left_side] = hsv[:, :, 2][left_side] * shadow_density
    else:
        hsv[:, :, 2][right_side] = hsv[:, :, 2][right_side] * shadow_density

    img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return img


for image in tqdm.tqdm(train_input_path_list):
    images_aug=cv2.imread(image)
    images_aug=np.array(images_aug).astype(np.uint8)
    
    new_image_path=image[:-4]+"-1"+".jpg"
    new_image_path=new_image_path.replace('image', 'augmentation')
    #images_aug=transformed_image_1 = transform(image=images_aug)['image']
    images_aug=augment_shadow(images_aug)
    #images_aug = seq1_2(images=images_aug)
    
    #images_aug= am.brighten(images_aug[0:1024],brightness_coeff=(1))
    #images_aug= am.random_brightness(images_aug) 
    cv2.imwrite(new_image_path,images_aug)


for mask_name in tqdm.tqdm(mask_list):
   mask_path = os.path.join(MASK_DIR, mask_name)
   mask_aug = cv2.imread(mask_path)
   mask_aug=augment_shadow(mask_aug)
   mask_aug = np.array(mask_aug)
   aug_mask_path = mask_path.replace('masks', 'shadow_mask')
   aug_mask_path = aug_mask_path.replace('cfc', 'shadow_cfc')
   cv2.imwrite(aug_mask_path, mask_aug)


for image_name in tqdm.tqdm(image_list):
    image_path = os.path.join(IMAGE_DIR, image_name)
    image = Image.open(image_path)
    aug_color_img = T.ColorJitter(brightness=0.4, contrast=0.3, hue=0.06)
    image_aug = aug_color_img(image)
    image_aug = np.array(image_aug)
    aug_img_path = image_path.replace('images', 'aug_color_images2')
    aug_img_path = aug_img_path.replace('cfc', 'color_cfc')
    cv2.imwrite(aug_img_path, image_aug)
    



for mask_name in tqdm.tqdm(mask_list):
    mask_path = os.path.join(MASK_DIR, mask_name)
    mask = Image.open(mask_path)
    #aug_color_mask = T.ColorJitter(brightness=0.8, contrast=0.0, hue=0.06)
    #mask_aug = aug_color_mask(mask)
    mask_aug = np.array(mask)
    aug_mask_path = mask_path.replace('masks', 'aug_color_masks')
    aug_mask_path = aug_mask_path.replace('cfc', 'color_cfc')
    cv2.imwrite(aug_mask_path, mask_aug)




image_list = os.listdir(IMAGE_DIR)
mask_list = os.listdir(MASK_DIR)

for image_name in tqdm.tqdm(image_list):
    image_path = os.path.join(IMAGE_DIR, image_name)
    #mask_path = os.path.join(MASK_DIR, image_name)
    
    image = cv2.imread(image_path)
    image_flipLR = np.fliplr(image)
    
    #mask = cv2.imread(mask_path)
    #mask_flipLR = np.fliplr(mask)
    
    #sigma = 0.155
    #noisy_random_image2 = random_noise(image_flipLR, var=sigma**2)
    #print(noisy_random_image2)
    #noisy_random_mask = random_noise(mask_flipLR, var=sigma**2)

    aug_img_path = image_path.replace('images', 'aug_fliplr_images')
    aug_img_path = aug_img_path.replace('cfc', 'aug_cfc')
    #aug_mask_path = mask_path.replace('masks', 'aug_masks')
    
    cv2.imwrite(aug_img_path, image_flipLR)
    #cv2.imwrite(aug_mask_path, noisy_random_mask)
    

for mask_name in tqdm.tqdm(mask_list):
    mask_path = os.path.join(MASK_DIR, mask_name)
    mask = cv2.imread(mask_path)
    mask_flipLR = np.fliplr(mask)
    aug_mask_path = mask_path.replace('masks', 'aug_fliplr_masks')
    aug_mask_path = aug_mask_path.replace('cfc', 'aug_cfc')
    
    cv2.imwrite(aug_mask_path, mask_flipLR)
    
