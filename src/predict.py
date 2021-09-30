from model import UNet
from preprocess import tensorize_image, tensorize_mask, image_mask_check
import os
import glob
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tqdm
import cv2

######### PARAMETERS ##########
valid_size = 0.3
test_size  = 0.1
batch_size = 2
epochs = 20
cuda = False
input_shape = (224, 224)
n_classes = 2
###############################

######### DIRECTORIES #########
SRC_DIR = os.getcwd()
ROOT_DIR = os.path.join(SRC_DIR, '..')
DATA_DIR = os.path.join(ROOT_DIR, 'data')
IMAGE_DIR = os.path.join(DATA_DIR, 'images')
MASK_DIR = os.path.join(DATA_DIR, 'masks')
###############################


# PREPARE IMAGE AND MASK LISTS
image_path_list = glob.glob(os.path.join(IMAGE_DIR, '*'))
image_path_list.sort()

mask_path_list = glob.glob(os.path.join(MASK_DIR, '*'))
mask_path_list.sort()




# DATA CHECK
image_mask_check(image_path_list, mask_path_list)

test_input_path_list = image_path_list
test_label_path_list = mask_path_list

    
#model = torch.load('../data/model/model.pth')
model = torch.load('../data/model/model.pth', map_location=torch.device('cpu'))
model.eval()


correct_predict = 0
wrong_predict = 0
for i in tqdm.tqdm(range(len(test_input_path_list))):
    batch_test = test_input_path_list[i:i+1]
    test_input = tensorize_image(batch_test, input_shape, cuda)
    with torch.no_grad():
        outs = model(test_input)
    out = torch.argmax(outs, axis=1)
    out_cpu = out.cpu()
    outputs_list = out_cpu.detach().numpy()
    mask = np.squeeze(outputs_list, axis=0)
    
    batch_label = test_label_path_list[i:i+1]
    test_label = tensorize_mask(batch_label, input_shape, n_classes, cuda)
    out_test_acc = (outs > 0.5).float()
    correct_predict += (out_test_acc == test_label).float().sum().item()
    wrong_predict += (out_test_acc != test_label).float().sum().item()
    
    img = cv2.imread(batch_test[0])
    #mg = cv2.resize(img, (224,224))
    mask = cv2.resize(mask.astype(np.uint8), (1920,1208))
    mask_ind = mask == 1
    cpy_img = img.copy()
    img[mask==1,:] = (255,0,125)
    opac_image = (img/2+cpy_img/2).astype(np.uint8)
    predict_name = batch_test[0]
    predict_path = predict_name.replace('images', './predict')
    #print(predict_path)
    #predict_path = "./predict"
    cv2.imwrite(predict_path,opac_image.astype(np.uint8))
    
total_predict = wrong_predict + correct_predict
accuracy_test = 100 * correct_predict / total_predict
print('Testing Accuracy:', accuracy_test)

