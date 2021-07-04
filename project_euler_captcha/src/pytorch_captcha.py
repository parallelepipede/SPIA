#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  5 19:39:13 2021

@author: paul
"""

from captcha.image import ImageCaptcha
import os
import itertools
import uuid
import shutil
import numpy as np
import cv2
from random import random
import sys
import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
#from torch.utils.tensorboard import SumaryWriter

PATH = os.getcwd()
WIDTH, HEIGHT = 135,50

def gen_captcha_dataset(width, height, img_dir,iteration,num_letters,replace = False):
    if replace and os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    image  = ImageCaptcha(width = width, height = height)
    for c in range(iteration):
        print('Generating {}/{}'.format(c,iteration))
        for i in itertools.permutations([str(char) for char in range(10)],num_letters):
            captcha = ''.join(i)
            image.write(captcha,os.path.join(img_dir,'{}_{}.png'.format(captcha,uuid.uuid4())))
            
#gen_captcha_dataset(WIDTH,HEIGHT,'five_numbers',1,5)
            
def load_data(path,num_letters,test_rate = .1):
    print('Loading dataset ...')
    y_train,x_train,y_test,x_test = [],[],[],[]
    
    # r = root, d = directories, f = files
    counter = 0
    for r,d,f in os.walk(path):
        for file in f[:100]: 
            
            if file.endswith('.png'):
                numbers_in = file.split('_')[0]
                if len(numbers_in) != num_letters : 
                    continue
                counter += 1
                label = np.zeros((num_letters,10))
                for i in range(num_letters):
                    label[i,int(numbers_in[i])] = 1
                img = cv2.imread(os.path.join(r,file))
                img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img,(WIDTH//2,HEIGHT//2),interpolation = cv2.INTER_AREA)
                img = np.reshape(img,(1,img.shape[0],img.shape[1]))
                
                if random() < test_rate :
                    y_test.append(label)
                    x_test.append(img)
                else:
                    y_train.append(label)
                    x_train.append(img)
    print('Dataset size : {}, (train : {}, test : {})'.format(counter, len(y_train),len(y_test)))
    return np.array(x_train).astype('float32')/255,np.array(y_train),np.array(x_test).astype('float32')/255,np.array(y_test)

x_train,y_train,x_test,y_test = load_data('data',5)

print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
num_letters = 5
s_train = [x_train[:,i,:] for i in range(num_letters)]
s_test = [x_test[:,i,:] for i in range(num_letters)]

save_dir = os.path.join(PATH,'saved_models')

class Net(nn.Module):
    def __init__(self):
        super(Net,self).__init__()
        self.conv1 = nn.Conv2d(1,32,kernel_size = 5)
        self.pool  = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(32,48,kernel_size = 5)
        self.conv3 = nn.Conv2d(48,64,kernel_size = 5)
        self.dropout = nn.Dropout(p = .3)
        self.fc1 = nn.Linear(64*31*31,512)
        self.fc2 = nn.Linear(512,10)
        self.epochs = 1
        
    def forward(self,x):
        print(x.shape, 'x')
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x))) 
        x = self.dropout(x)
        x = x.view(x.size(0),-1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        out = [(self.fc2(x),i) for i in range(num_letters)]
        return out
        #x = F.relu(self.fc2(x))
 
net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr = 10**-2)
    
def train():
    print("Start training")
    for epoch in range(net.epochs):
        for data,target in train_loader:#zip(x_train,y_train):
            #print(target)
            optimizer.zero_grad()
            outputs = net(data)
            #print(outputs, "output")
            sys.exit(0)
            loss = 0
            for out in outputs : 
                loss += criterion(out,target)
            loss.backward()
            optimizer.step()
        if epoch % 10 == 0: 
            print(" [{}/{} ({:.0f}%)]\tLoss: {:.6f}".format(
                epoch,
                net.epochs,
                100.0 * epoch / net.epochs,
                loss.item(),
            ))
            
import sklearn.model_selection
#x_train,x_test,y_train,y_test = sklearn.model_selection.train_test_split(x_train,y_train,test_size = 1/10,shuffle = True)
training_dataset = torch.utils.data.TensorDataset(torch.Tensor(x_train),torch.Tensor(y_train))
train_loader = torch.utils.data.DataLoader(training_dataset, batch_size= 128,shuffle = True)

train()
