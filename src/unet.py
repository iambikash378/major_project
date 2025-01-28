import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu

class UNet(nn.Module):

    def __init__(self):
        super().__init__()
        
        #Encoder

        # Input = 128*128*4
        self.conv1_1 = nn.Conv2d(4, 64, kernel_size = 3, padding = 1) # Output = 128*128*64
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size = 3, padding = 1) # Output = 128*128*64
        self.pool1 = nn.MaxPool2d(kernel_size = 2, stride = 2) #Output = 64*64*64

        #Input = 64*64*64
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size = 3, padding = 1) # Output = 64*64*128
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size = 3, padding = 1) # Output = 64*64*128
        self.pool2 = nn.MaxPool2d(kernel_size = 2 , stride = 2) #Output = 32*32*128

        #Input = 32*32*128
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size = 3, padding = 1) #Output = 32*32*256
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size = 3, padding = 1) #Output = 32*32*256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride = 2) #Output = 16*16*256

        #Input = 16*16*256
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size = 3, padding = 1) #Output = 16*16*512
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size = 3, padding = 1) #Output = 16*16*512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride = 2) #Output = 8*8*512

        #Input = 8*8*512
        self.conv5_1 = nn.Conv2d(512, 1024, kernel_size = 3, padding = 1) #Output = 8*8*1024
        self.conv5_2 = nn.Conv2d(1024, 1024, kernel_size = 3, padding = 1) #Output = 8*8*1024

        #Decoder

        #Input = 8*8*1024
        self.upconv1 = nn.ConvTranspose2d(1024,512,kernel_size = 2, stride = 2)
        self.conv1_i = nn.Conv2d(1024,512, kernel_size = 3, padding = 1)
        self.conv1_ii = nn.Conv2d(512, 512, kernel_size = 3, padding = 1)

        self.upconv2 = nn.ConvTranspose2d(512,256, kernel_size = 2, stride = 2)
        self.conv2_i = nn.Conv2d(512, 256, kernel_size = 3, padding = 1)
        self.conv2_ii = nn.Conv2d(256, 256, kernel_size =3, padding = 1)

        self.upconv3 = nn.ConvTranspose2d(256,128, kernel_size = 2, stride = 2)
        self.conv3_i = nn.Conv2d(256, 128, kernel_size = 3, padding = 1)
        self.conv3_ii = nn.Conv2d(128, 128, kernel_size =3, padding = 1)

        self.upconv4 = nn.ConvTranspose2d(128,64, kernel_size = 2, stride = 2)
        self.conv4_i = nn.Conv2d(128, 64, kernel_size = 3, padding = 1)
        self.conv4_ii = nn.Conv2d(64, 64, kernel_size =3, padding = 1)

        self.outconv = nn.Conv2d(64,1, kernel_size = 1)


    def forward(self, x):
        #Encoder
        relu1_1 = relu(self.conv1_1(x))
        relu1_2 = relu(self.conv1_2(relu1_1))
        pooled1 = self.pool1(relu1_2)

        relu2_1 = relu(self.conv2_1(pooled1))
        relu2_2 = relu(self.conv2_2(relu2_1))
        pooled2 = self.pool2(relu2_2)
        
        relu3_1 = relu(self.conv3_1(pooled2))
        relu3_2 = relu(self.conv3_2(relu3_1))
        pooled3 = self.pool3(relu3_2)

        relu4_1 = relu(self.conv4_1(pooled3))
        relu4_2 = relu(self.conv4_2(relu4_1))
        pooled4 = self.pool4(relu4_2)

        relu5_1 = relu(self.conv5_1(pooled4))
        relu5_2 = relu(self.conv5_2(relu5_1))

        #Decoder
        up1 = self.upconv1(relu5_2)
        concat1 = torch.cat([up1, relu4_2], dim = 1)
        relu1_i = relu(self.conv1_i(concat1))
        relu1_ii = relu(self.conv1_ii(relu1_i))

        up2 = self.upconv2(relu1_ii)
        concat2 = torch.cat([up2, relu3_2], dim = 1)
        relu2_i = relu(self.conv2_i(concat2))
        relu2_ii = relu(self.conv2_ii(relu2_i))       

        up3 = self.upconv3(relu2_ii)
        concat3 = torch.cat([up3, relu2_2], dim = 1)
        relu3_i = relu(self.conv3_i(concat3))
        relu3_ii = relu(self.conv3_ii(relu3_i))

        up4 = self.upconv4(relu3_ii)
        concat4 = torch.cat([up4, relu1_2], dim = 1)
        relu4_i = relu(self.conv4_i(concat4))
        relu4_ii = relu(self.conv4_ii(relu4_i))

        #Output Layer

        out = self.outconv(relu4_ii)

        return out


