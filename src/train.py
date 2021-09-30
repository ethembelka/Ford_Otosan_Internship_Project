#from model import FoInternNet
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
import matplotlib.pyplot as plt
from random import shuffle

######### PARAMETERS ##########
valid_size = 0.15
test_size  = 0.15
batch_size = 10
epochs = 15
cuda = True
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
DATA_DIR = "../input/forstajp1/p1/p1/intern-p1/data"
IMAGE_DIR = "../input/forstajp1/p1/p1/intern-p1/data/images"
MASK_DIR = "../input/forstajp1/masks/masks"
AUG_FLIPLR_IMAGE_DIR = "../input/forstajp1/aug_fliplr_data/aug_fliplr_data/aug_fliplr_images"
AUG_FLIPLR_MASK_DIR = "../input/forstajp1/aug_fliplr_data/aug_fliplr_data/aug_fliplr_masks"
AUG_COLOR_IMAGE_DIR = "../input/forstajp1/aug_color_data/aug_color_data/aug_color_images"
AUG_COLOR_MASK_DIR = "../input/forstajp1/aug_color_data/aug_color_data/aug_color_masks"
AUG_SHADOW_IMAGE_DIR = "../input/forstajp1/shadow_img/shadow_img"
AUG_SHADOW_MASK_DIR = "../input/forstajp1/shadow_mask/shadow_mask"


# PREPARE IMAGE AND MASK LISTS
image_path_list1 = glob.glob(os.path.join(IMAGE_DIR, '*'))
image_path_list1.sort()

mask_path_list1 = glob.glob(os.path.join(MASK_DIR, '*'))
mask_path_list1.sort()

#here
aug_fliplr_image_path_list = glob.glob(os.path.join(AUG_FLIPLR_IMAGE_DIR, '*'))
aug_fliplr_image_path_list.sort()

aug_fliplr_mask_path_list = glob.glob(os.path.join(AUG_FLIPLR_MASK_DIR, '*'))
aug_fliplr_mask_path_list.sort()

aug_color_image_path_list = glob.glob(os.path.join(AUG_COLOR_IMAGE_DIR, '*'))
aug_color_image_path_list.sort()

aug_color_mask_path_list = glob.glob(os.path.join(AUG_COLOR_MASK_DIR, '*'))
aug_color_mask_path_list.sort()

aug_shadow_image_path_list = glob.glob(os.path.join(AUG_SHADOW_IMAGE_DIR, '*'))
aug_shadow_image_path_list.sort()

aug_shadow_mask_path_list = glob.glob(os.path.join(MASK_DIR, '*'))
aug_shadow_mask_path_list.sort()
#here

image_path_list = image_path_list1 + aug_color_image_path_list + aug_fliplr_image_path_list + aug_shadow_image_path_list
mask_path_list = mask_path_list1 + aug_color_mask_path_list + aug_fliplr_mask_path_list + aug_shadow_mask_path_list

# DATA CHECK
N = len(image_path_list)
ind_list = [i for i in range(N)]
shuffle(ind_list)
image_path_list  = list(np.array(image_path_list)[ind_list])
mask_path_list = list(np.array(mask_path_list)[ind_list])

image_mask_check(image_path_list, mask_path_list)


# SHUFFLE INDICES
indices = np.random.permutation(len(image_path_list))

# DEFINE TEST AND VALID INDICES
test_ind  = int(len(indices) * test_size)
valid_ind = int(test_ind + len(indices) * valid_size)

# SLICE TEST DATASET FROM THE WHOLE DATASET
test_input_path_list = image_path_list[:test_ind]
test_label_path_list = mask_path_list[:test_ind]

# SLICE VALID DATASET FROM THE WHOLE DATASET
valid_input_path_list = image_path_list[test_ind:valid_ind]
valid_label_path_list = mask_path_list[test_ind:valid_ind]

# SLICE TRAIN DATASET FROM THE WHOLE DATASET
train_input_path_list = image_path_list[valid_ind:]
train_label_path_list = mask_path_list[valid_ind:]



# DEFINE STEPS PER EPOCH
steps_per_epoch = len(train_input_path_list)//batch_size

# CALL MODEL
#model = FoInternNet(input_size=input_shape, n_classes=2)

#model = UNet(input_size=input_shape, n_classes=2)
model = torch.load('../input/model15/models15.pth')

# DEFINE LOSS FUNCTION AND OPTIMIZER
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay = 1e-6)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', verbose=True, patience=2)


# IF CUDA IS USED, IMPORT THE MODEL INTO CUDA
if cuda:
    model = model.cuda()
    criterion = criterion.cuda()

train_losses=[]
valid_losses=[]
train_accuracies=[]
valid_accuracies=[]
# TRAINING THE NEURAL NETWORK
for epoch in range(epochs):
    running_loss = 0
    
    correct_train = 0
    wrong_train = 0
    correct_valid = 0
    wrong_valid = 0
    for ind in tqdm.tqdm(range(steps_per_epoch)):
        batch_input_path_list = train_input_path_list[batch_size*ind:batch_size*(ind+1)]
        batch_label_path_list = train_label_path_list[batch_size*ind:batch_size*(ind+1)]
        batch_input = tensorize_image(batch_input_path_list, input_shape, cuda)
        batch_label = tensorize_mask(batch_label_path_list, input_shape, n_classes, cuda)

        optimizer.zero_grad()
        
        #print(batch_input.shape)
        #batch_input,_ = torch.max(batch_input,1,keepdim=True)
        #print(batch_input.shape)
        
        outputs = model(batch_input)
        
        output = (outputs > 0.5).float()
        correct_train += (output == batch_label).float().sum().item()
        wrong_train += (output != batch_label).float().sum().item()
        #print("correct:", correct_train)
        #print("wrong", wrong_train)
        total_train = wrong_train + correct_train
        
        #batch_label,_ = torch.max(batch_label,1,keepdim=True)
        #print(outputs)
        loss = criterion(outputs, batch_label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        #
        #print(ind)
        if ind == steps_per_epoch-1:
            print('training loss on epoch {}: {}'.format(epoch, running_loss/ steps_per_epoch))
            train_losses.append(running_loss/steps_per_epoch)
            model.eval()
            val_loss = 0
            for (valid_input_path, valid_label_path) in zip(valid_input_path_list, valid_label_path_list):
                batch_input = tensorize_image([valid_input_path], input_shape, cuda)
                batch_label = tensorize_mask([valid_label_path], input_shape, n_classes, cuda)
                with torch.no_grad():
                    outputs = model(batch_input)
                out_acc = (outputs > 0.5).float()
                correct_valid += (out_acc == batch_label).float().sum().item()
                wrong_valid += (out_acc != batch_label).float().sum().item()
                total_valid = wrong_valid + correct_valid 
                
                loss = criterion(outputs, batch_label)
                val_loss += loss.item()
            valid_losses.append(val_loss / len(valid_input_path_list))
            scheduler.step(val_loss / len(valid_input_path_list))
            model.train()
            print('validation loss on epoch {}: {}'.format(epoch, val_loss / len(valid_input_path_list)))
            
    accuracy_train = 100 * correct_train / total_train
    train_accuracies.append(accuracy_train)
    print('Training Accuracy on epoch {}: {}'.format(epoch,accuracy_train))
    accuracy_valid = 100 * correct_valid / total_valid
    valid_accuracies.append(accuracy_valid)
    print('Validation Accuracy on epoch {}: {}'.format(epoch,accuracy_valid))
    model_name = './models' + str(epoch) + '.pth'
    torch.save(model, model_name)
            
def graph_losses(epochs, train_losses, valid_losses):
    epochs_num = list(range(1,epochs+1,1))
    plt.plot(epochs_num, train_losses, 'r', label='Training Loss')
    plt.plot(epochs_num, valid_losses, 'y', label='Valdation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig(fname="./loss_graph.png", facecolor='green')
    plt.show()           

graph_losses(epochs, train_losses, valid_losses)

torch.save(model, "./torch_saved/modell.pth" )
print("Torch Saved")
modell = torch.load("./torch_saved/modell.pth")

def graph_accuracies(epochs, train_accuracies, valid_accuracies):
    epochs_num = list(range(1,epochs+1,1))
    plt.plot(epochs_num,train_accuracies, 'r', label='Training Accuracy')
    plt.plot(epochs_num,valid_accuracies,'y', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(fname='./accuracy_grapg.png', facecolor='green')
    plt.show()
    
graph_accuracies(epochs, train_accuracies, valid_accuracies)

def graph_all(epochs, train_losses, train_accuracies, valid_losses, valid_accuracies):
    epochs_num = list(range(1,epochs+1,1))
    plt.plot(epochs_num, train_losses, 'r', label='Training Loss')
    plt.plot(epochs_num, train_accuracies, 'g', label='Training Accuracy')
    plt.plot(epochs_num, valid_losses, 'b', label='Validation Loss')
    plt.plot(epochs_num, valid_accuracies, 'y', label='Validation Accuracy')
    plt.title('Training, Validation Loss and Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Loss and Accuracy')
    plt.legend()
    plt.savefig(fname='./all_graph.png')
    plt.show()

graph_all(epochs, train_losses, train_accuracies, valid_losses, valid_accuracies)

def predict(test_input_path_list, test_label_path_list):
    correct_predict = 0
    wrong_predict = 0
    for i in tqdm.tqdm(range(len(test_input_path_list))):
        batch_test = test_input_path_list[i:i+1]
        test_input = tensorize_image(batch_test, input_shape, cuda)
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
        mg = cv2.resize(img, (224,224))
        mask_ind = mask == 1
        cpy_img = mg.copy()
        mg[mask==1,:] = (255,0,125)
        opac_image = (mg/2+cpy_img/2).astype(np.uint8)
        predict_name = batch_test[0]
        predict_path = predict_name.replace('../input/forstajp1/p1/p1/intern-p1/data/images', './predict')
        #print(predict_path)
        #predict_path = "./predict"
        cv2.imwrite(predict_path,opac_image.astype(np.uint8))
        
    total_predict = wrong_predict + correct_predict
    accuracy_test = 100 * correct_predict / total_predict
    print('Testing Accuracy:', accuracy_test)

predict(test_input_path_list, test_label_path_list)