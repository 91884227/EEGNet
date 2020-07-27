#!/usr/bin/env python
# coding: utf-8

# In[11]:


import torch
import torch.nn as nn


# In[13]:


class EEG_Net(nn.Module):
    def __init__(self, actfun_ = "ELU"):
        super(EEG_Net, self).__init__()
        
        if( actfun_ == "ELU"):
            self.act_func = nn.ELU(alpha = 1.0)
            
        elif( actfun_ == "ReLU" ):
            self.act_func = nn.ReLU()
            
        elif( actfun_ == "LeakyReLU" ):
            self.act_func = nn.LeakyReLU(0.2, inplace=True) 
        else:
            assert False, 'wrong actfun_' 
        
        # first conv
        self.conv1 = nn.Conv2d(in_channels = 1, 
                               out_channels = 16, 
                               kernel_size=(1, 51), 
                               stride = (1, 1), 
                               padding = (0, 25), 
                               bias = False)
        self.batch_norm1 = nn.BatchNorm2d(num_features = 16,
                                          eps = 1e-05, 
                                          momentum = 0.1, 
                                          affine = True, 
                                          track_running_stats = True)
        
        # depthwise conv
        self.conv2 = nn.Conv2d(in_channels = 16, 
                               out_channels = 32, 
                               kernel_size=(2, 1), 
                               stride = (1, 1), 
                               groups = 16, 
                               bias = False)
        self.batch_norm2 = nn.BatchNorm2d(num_features = 32,
                                          eps = 1e-05, 
                                          momentum = 0.1, 
                                          affine = True, 
                                          track_running_stats = True)
        # self.ELU2 = nn.ELU(alpha = 1.0)
        self.pooling2 = nn.AvgPool2d(kernel_size = (1, 4), 
                                     stride = (1, 4), 
                                     padding = 0)
        self.dropout2 = nn.Dropout(p = 0.25)
        
        # separable conv
        self.conv3 = nn.Conv2d(in_channels = 32, 
                               out_channels = 32, 
                               kernel_size=(1, 15), 
                               stride = (1, 1), 
                               padding = (0, 7), 
                               bias = False)
        self.batch_norm3 = nn.BatchNorm2d(num_features = 32,
                                          eps = 1e-05, 
                                          momentum = 0.1, 
                                          affine = True, 
                                          track_running_stats = True) 
        # self.ELU3 = nn.ELU(alpha = 1.0)
        self.pooling3 = nn.AvgPool2d(kernel_size = (1, 8), 
                                     stride = (1, 8), 
                                     padding = 0)
        self.dropout3 = nn.Dropout(p = 0.25)
        
        # classify
        self.linear4 = nn.Linear(in_features = 736, 
                                 out_features = 2, 
                                 bias = True)
    def forward(self, x):
        x = self.conv1(x)
        x = self.batch_norm1(x)
        
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.act_func(x)
        x = self.pooling2(x)
        x = self.dropout2(x)
        
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.act_func(x)
        x = self.pooling3(x)
        x = self.dropout3(x)
        
        x = x.view(-1, 736)
        x = self.linear4(x)
        return(x)

class DeepConvNet(nn.Module):
    def __init__(self, actfun_ = "ELU"):
        super(DeepConvNet, self).__init__()

        if( actfun_ == "ELU"):
            self.act_func = nn.ELU(alpha = 1.0)

        elif( actfun_ == "ReLU" ):
            self.act_func = nn.ReLU()

        elif( actfun_ == "LeakyReLU" ):
            self.act_func = nn.LeakyReLU(0.2, inplace=True) 
        else:
            assert False, 'wrong actfun_' 
            
        # 1
        self.conv1 = nn.Conv2d(in_channels = 1, out_channels = 25, kernel_size = (1, 5))
        
        # 2
        self.conv2 = nn.Conv2d(in_channels = 25, out_channels = 25, kernel_size = (2, 1))
        self.batch_norm2 = nn.BatchNorm2d(num_features = 25,
                                          eps = 1e-05, 
                                          momentum = 0.1)
        self.maxpool2 = nn.MaxPool2d((1, 2))
        self.dropout2 = nn.Dropout(p = 0.5)
        
        # 3
        self.conv3 = nn.Conv2d(in_channels = 25, out_channels = 50, kernel_size = (1, 5))
        self.batch_norm3 = nn.BatchNorm2d(num_features = 50,
                                          eps = 1e-05, 
                                          momentum = 0.1)
        self.maxpool3 = nn.MaxPool2d((1, 2))
        self.dropout3 = nn.Dropout(p = 0.5)
        
        # 4
        self.conv4 = nn.Conv2d(in_channels = 50, out_channels = 100, kernel_size = (1, 5))       
        self.batch_norm4 = nn.BatchNorm2d(num_features = 100,
                                          eps = 1e-05, 
                                          momentum = 0.1)
        self.maxpool4 = nn.MaxPool2d((1, 2))        
        self.dropout4 = nn.Dropout(p = 0.5)    

        # 5
        self.conv5 = nn.Conv2d(in_channels = 100, out_channels = 200, kernel_size = (1, 5))       
        self.batch_norm5 = nn.BatchNorm2d(num_features = 200,
                                          eps = 1e-05, 
                                          momentum = 0.1)
        self.maxpool5 = nn.MaxPool2d((1, 2))        
        self.dropout5 = nn.Dropout(p = 0.5)
        
        # linear
        self.linear = nn.Linear(8600, 2)
        
    def forward(self, x):
        # 1
        x = self.conv1(x)
        # print(x.size())
        # 2
        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.act_func(x)
        x = self.maxpool2(x)
        x = self.dropout2(x)
        #print(x.size())
        # 3
        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = self.act_func(x)
        x = self.maxpool3(x)
        x = self.dropout3(x)
        #print(x.size())
        # 4
        x = self.conv4(x)
        x = self.batch_norm4(x)
        x = self.act_func(x)
        x = self.maxpool4(x)
        x = self.dropout4(x)
        #print(x.size())
        # 5
        x = self.conv5(x)
        x = self.batch_norm5(x)
        x = self.act_func(x)
        x = self.maxpool5(x)
        x = self.dropout5(x)   
        #print(x.size())
        # linear
        x = x.view(-1, 8600)
        x = self.linear(x)
        return(x)