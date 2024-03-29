import torch.nn as nn
import torch
import numpy as np
import string
import secrets
import difflib
import time
from scipy.signal import firwin, lfilter
import cv2

# 隐写算法
class SteganographyAlgorithm(nn.Module):
    def __init__(self):
        super(SteganographyAlgorithm, self).__init__()
        self.secretinformation = self.read_file()
        self.secvec = self.sec2vec(self.secretinformation)
        self.realusedlength = 0
        
    def read_file(self):
        secinffile = open('secret.txt', encoding='ascii')
        secinf = secinffile.read()
        secinffile.close()
        
        return secinf
    
    def sec2vec(self, secret):
        pass
        
    def inverse(self, arr):
        pass
    
    def write_secret(self, h, w):
        pass
    
    def get_realusedlength(self):
        return self.realusedlength
    
    def extract_secret(self, layer):
        pass

# GLU  门控线性单元Gated linear units
class GLU(nn.Module):
    def __init__(self):
        super(GLU, self).__init__()
        
    def forward(self, x,gated_x):
        return x * torch.sigmoid(gated_x)


class downSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(downSample, self).__init__()

        self.convLayer = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding),
                                       nn.InstanceNorm2d(num_features=out_channels,
                                                         affine=True))
        self.convLayer_gates = nn.Sequential(nn.Conv2d(in_channels=in_channels,
                                                       out_channels=out_channels,
                                                       kernel_size=kernel_size,
                                                       stride=stride,
                                                       padding=padding),
                                             nn.InstanceNorm2d(num_features=out_channels,
                                                               affine=True))
                                                               
        self.glu_layer = GLU()
    def forward(self, x):
        # GLU
        return self.glu_layer(self.convLayer(x),self.convLayer_gates(x))
        
        
class upSample(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(upSample, self).__init__()

        self.convLayer = nn.Sequential(nn.ConvTranspose2d(in_channels=in_channels,
                                                 out_channels=out_channels,
                                                 kernel_size=kernel_size,
                                                 stride=stride,
                                                 padding=padding),
                                       nn.InstanceNorm2d(num_features=out_channels,
                                                         affine=True))
        self.convLayer_gates = nn.Sequential(nn.ConvTranspose2d(in_channels=in_channels,
                                                       out_channels=out_channels,
                                                       kernel_size=kernel_size,
                                                       stride=stride,
                                                       padding=padding),
                                             nn.InstanceNorm2d(num_features=out_channels,
                                                               affine=True))
                                                               
        self.glu_layer = GLU()
    def forward(self, x):
        # GLU
        return self.glu_layer(self.convLayer(x),self.convLayer_gates(x))
        
class AfterProcess(nn.Module):
    def __init__(self):
        super(AfterProcess, self).__init__()
        self.AfterChunk_6 = downSample(32, 1, (3, 9), (1, 1), (1, 4))
        
        
        self.AfterChunk_5 = downSample(64, 1, (3, 9), (1, 1), (1, 4))
                                
        self.AfterChunk_4 = downSample(128, 1, (3, 9), (1, 1), (1, 4))

        self.AfterChunk_3 = downSample(256, 1, (3, 9), (1, 1), (1, 4))

        self.AfterChunk_2 = downSample(64, 1, (3, 9), (1, 1), (1, 4))

        self.AfterChunk_1 = downSample(5, 1, (3, 9), (1, 1), (1, 4))
    
    def forward(self,input,layer=5):
        if layer == 1:
            out = self.AfterChunk_1(input) 
        elif layer == 2:
            out = self.AfterChunk_2(input) 
        elif layer == 3:
            out = self.AfterChunk_3(input) 
        elif layer == 4:
            out = self.AfterChunk_4(input)
        elif layer == 5:
            out = self.AfterChunk_5(input) 
        
        return out
    

class GeneratorLeft(nn.Module):
    def __init__(self,n_spk):
        super(GeneratorLeft, self).__init__()
        self.n_spk = n_spk
        self.down_0 = None
        self.down_1 = None
        self.down_2 = None
        self.down_3 = None
        self.down_4 = None
        self.down_5 = None
        
        self.downSample_0 = downSample(in_channels=1,  # TODO 1 ?
                                out_channels=32,
                                kernel_size=(3, 9),
                                stride=(1, 1),
                                padding=(1, 4))
        
        self.downSample_1 = downSample(in_channels=32,  # TODO 1 ?
                                out_channels=64,
                                kernel_size=(3, 9),
                                stride=(1, 1),
                                padding=(1, 4))
                                
        self.downSample_2 = downSample(64,128,(4, 8),(2, 2),(1, 3))

        self.downSample_3 = downSample(128,256,(4, 8),(2, 2),(1, 3))

        self.downSample_4 = downSample(256,64,(3, 5),(1, 1),(1, 2))

        self.downSample_5 = downSample(64,5,(9, 5),(9, 1),(4, 2))
        
    def get_parameter(self,layer):
        if layer == 2:
            return self.down_2
        elif layer == 3:
            return self.down_3
        elif layer == 4:
            return self.down_4
        elif layer == 5:
            return self.down_5
        elif layer == 1:
            return self.down_1
        elif layer == 0:
            return self.down_0
    
    def forward(self,input):
        # 下采样
        self.down_0 = self.downSample_0(input) 
        self.down_1 = self.downSample_1(self.down_0)  
        self.down_2 = self.downSample_2(self.down_1)
        self.down_3 = self.downSample_3(self.down_2) 
        self.down_4 = self.downSample_4(self.down_3) 
        self.down_5 = self.downSample_5(self.down_4) 
        
        return self.down_5


class GeneratorRight(nn.Module):
    def __init__(self,n_spk):
        super(GeneratorRight, self).__init__()
        self.n_spk = n_spk
        self.Up_1 = None
        self.Up_2 = None
        self.Up_3 = None
        self.Up_4 = None
        self.Up_5 = None
        
        self.Up_1_length_h = 0
        self.Up_2_length_h = 0
        self.Up_3_length_h = 0
        self.Up_4_length_h = 0
        self.Up_5_length_h = 0
        
        self.Up_1_length_w = 0
        self.Up_2_length_w = 0
        self.Up_3_length_w = 0
        self.Up_4_length_w = 0
        self.Up_5_length_w = 0
        
        # 无隐写
        self.upSample_1 = upSample(5 + self.n_spk, 64, (9, 5), (9, 1), (0, 2))
        self.upSample_1_steg = upSample(5 + self.n_spk+1, 64, (9, 5), (9, 1), (0, 2))                           
                                   
        self.upSample_2 = upSample(64+ self.n_spk, 256, (3,5),(1,1),(1,2))
        self.upSample_2_steg = upSample(64+ self.n_spk+1, 256, (3,5),(1,1),(1,2))
                                   
        self.upSample_3 = upSample(256+ self.n_spk, 128, (4,8),(2,2),(1,3))   
        self.upSample_3_steg = upSample(256+ self.n_spk+1, 128, (4,8),(2,2),(1,3))                         
        
        self.upSample_4 = upSample(128+ self.n_spk, 64, (4,8),(2,2),(1,3))
        self.upSample_4_steg = upSample(128+ self.n_spk+1, 64, (4,8),(2,2),(1,3))
        
        self.upSample_5 = upSample(64+ self.n_spk, 32, (3,9),(1,1),(1,4))
        self.upSample_5_steg = upSample(64+ self.n_spk +1, 32, (3,9),(1,1),(1,4))

        self.deCNN = nn.ConvTranspose2d(in_channels=32 + self.n_spk,
                                        out_channels=1,
                                        kernel_size=(3,9),
                                        stride=(1,1),
                                        padding=(1,4))
    
    def get_write_length(self,layer):
        if layer == 1:
            return self.Up_1_length_h, self.Up_1_length_w
        elif layer == 2:
            return self.Up_2_length_h, self.Up_2_length_w
        elif layer == 3:
            return self.Up_3_length_h, self.Up_3_length_w
        elif layer == 4:
            return self.Up_4_length_h, self.Up_4_length_w
        elif layer == 5:
            return self.Up_5_length_h, self.Up_5_length_w
    
    def get_parameter(self,layer):
        if layer == 1:
            return self.Up_1
        elif layer == 2:
            return self.Up_2
        elif layer == 3:
            return self.Up_3
        elif layer == 4:
            return self.Up_4
        elif layer == 5:
            return self.Up_5
        
    def forward(self,input,lab,steg,sec = None, layer = None):
        
        c = lab.view(lab.size(0), lab.size(1), 1, 1) 
        
        c1 = c.repeat(1, 1, input.size(2), input.size(3)) 
        in_Up_1 = torch.cat([input, c1], dim=1) 
        self.Up_1_length_h = input.size(2)
        self.Up_1_length_w = input.size(3)
        if steg == False or (steg == True and layer != 1):
            self.Up_1 = self.upSample_1(in_Up_1)         
        elif steg == True and layer == 1:
            in_Up_1 = torch.cat([in_Up_1, sec], dim=1) 
            self.Up_1 = self.upSample_1_steg(in_Up_1)       
        
        c2 = c.repeat(1, 1, self.Up_1.size(2), self.Up_1.size(3))
        in_Up_2 = torch.cat([self.Up_1, c2], dim=1) 
        self.Up_2_length_h = self.Up_1.size(2)
        self.Up_2_length_w = self.Up_1.size(3)
        if steg == False or (steg == True and layer != 2):
            self.Up_2 = self.upSample_2(in_Up_2)        
        elif steg == True and layer == 2:
            in_Up_2 = torch.cat([in_Up_2, sec], dim=1) 
            self.Up_2 = self.upSample_2_steg(in_Up_2)         
        
        c3 = c.repeat(1, 1, self.Up_2.size(2), self.Up_2.size(3))
        in_Up_3 = torch.cat([self.Up_2, c3], dim=1)  
        self.Up_3_length_h = self.Up_2.size(2)
        self.Up_3_length_w = self.Up_2.size(3)
        if steg == False or (steg == True and layer != 3):
            self.Up_3 = self.upSample_3(in_Up_3)       
        elif steg == True and layer == 3:
            in_Up_3 = torch.cat([in_Up_3, sec], dim=1) 
            self.Up_3 = self.upSample_3_steg(in_Up_3)     
        
        
        c4 = c.repeat(1, 1, self.Up_3.size(2), self.Up_3.size(3))
        in_Up_4 = torch.cat([self.Up_3, c4], dim=1)  
        self.Up_4_length_h = self.Up_3.size(2)
        self.Up_4_length_w = self.Up_3.size(3)
        if steg == False or (steg == True and layer != 4):
            self.Up_4 = self.upSample_4(in_Up_4)         
        elif steg == True and layer == 4:
            in_Up_4 = torch.cat([in_Up_4, sec], dim=1)
            self.Up_4 = self.upSample_4_steg(in_Up_4)     
        
        c5 = c.repeat(1, 1, self.Up_4.size(2), self.Up_4.size(3))
        in_Up_5 = torch.cat([self.Up_4, c5], dim=1)
        self.Up_5_length_h = self.Up_4.size(2)
        self.Up_5_length_w = self.Up_4.size(3)
        if steg == False or (steg == True and layer != 5):
            self.Up_5 = self.upSample_5(in_Up_5)         
        elif steg == True and layer == 5:
            in_Up_5 = torch.cat([in_Up_5, sec], dim=1) 
            self.Up_5 = self.upSample_5_steg(in_Up_5)       
          
        c6 = c.repeat(1, 1, self.Up_5.size(2), self.Up_5.size(3))
        in_Up_6 = torch.cat([self.Up_5, c6], dim=1) 
        out = self.deCNN(in_Up_6)              
        
        return out