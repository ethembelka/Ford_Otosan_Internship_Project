import torch
import torch.nn as nn
import torch.nn.functional as F
from constant import *

def double_conv(in_c, out_c):
    conv = nn.Sequential(
        nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=out_c, out_channels=out_c, kernel_size = 3, padding=1),
        nn.BatchNorm2d(out_c),
        nn.ReLU(inplace=True),
        )
    return conv

def crop_img(tensor, target_tensor):
    target_size = target_tensor.size()[2]
    tensor_size = tensor.size()[2]
    delta1 = tensor_size-target_size
    delta2 = delta1 // 2
    if delta1 % 2 == 0:
        
        return tensor[:,:,delta2:tensor_size-delta2, delta2:tensor_size-delta2]
    else:
        return tensor[:,:,delta2:tensor_size-delta2-1, delta2:tensor_size-delta2-1]


class UNet(nn.Module):
    
    def __init__(self,input_size, n_classes):
        super(UNet, self).__init__()
        self.input_size = input_size
        self.n_classes = n_classes
        
        self.max_pool_2x2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.up_conv_1_trans = nn.ConvTranspose2d(in_channels=1024, out_channels=512, kernel_size=2, stride=2)
        self.up_conv_2_trans = nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size=2, stride=2)
        self.up_conv_3_trans = nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size=2, stride=2)
        self.up_conv_4_trans = nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=2, stride=2)
        
        self.down_conv_1 = double_conv(3, 64)
        self.down_conv_2 = double_conv(64, 128)
        self.down_conv_3 = double_conv(128, 256)
        self.down_conv_4 = double_conv(256, 512)
        self.down_conv_5 = double_conv(512, 1024)
        
        self.up_conv_1 = double_conv(1024, 512)
        self.up_conv_2 = double_conv(512, 256)
        self.up_conv_3 = double_conv(256, 128)
        self.up_conv_4 = double_conv(128, 64)
        self.out_conv = nn.Conv2d(64, n_classes, kernel_size=1,stride=1)
    
    def forward(self, image):
        
        #encoder
        x1 = self.down_conv_1(image)
        #print("x1",x1.size())
        x2 = self.max_pool_2x2(x1)
        x3 = self.down_conv_2(x2)
        x4 = self.max_pool_2x2(x3)
        x5 = self.down_conv_3(x4)
        x6 = self.max_pool_2x2(x5)
        x7 = self.down_conv_4(x6)
        #print("x7", x7.size())
        x8 = self.max_pool_2x2(x7)
        x9 = self.down_conv_5(x8)
        #print("x9",x9.size())
        
        #decoder
        x = self.up_conv_1_trans(x9)
        #print("x",x.size())
        y = crop_img(x7, x)
        x = self.up_conv_1(torch.cat([x,y],1))
        #print("x7", x7.size())
        #print("y", y.size())
        
        x = self.up_conv_2_trans(x)
        y = crop_img(x5, x)
        x = self.up_conv_2(torch.cat([x,y],1))
        
        x = self.up_conv_3_trans(x)
        y = crop_img(x3, x)
        x = self.up_conv_3(torch.cat([x,y],1))
        
        x = self.up_conv_4_trans(x)
        y = crop_img(x1, x)
        x = self.up_conv_4(torch.cat([x,y],1))
        
        x = self.out_conv(x)
        #print("xlast", x.size())
        result = nn.Softmax(dim=1)(x)
        return result


if __name__ == '__main__':
    #model = FoInternNet(input_size=(HEIGHT, WIDTH), n_classes=N_CLASS)
    
    
    image = torch.rand((1,3,224,224))
    model = UNet((224,224),2)
    print(model(image).size())

