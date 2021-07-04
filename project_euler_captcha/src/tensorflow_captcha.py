#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 24 10:23:32 2021

@author: paul
"""

import argparse
import os
import shutil
import uuid
from captcha.image import ImageCaptcha

import itertools
import sys
import cv2
import numpy as np
from random import random
import tensorflow
from tensorflow.keras.models import Model,load_model
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input
#import matplotlib.pyplot as plt

from selenium import webdriver
import time
import requests


PATH = os.getcwd()
WIDTH, HEIGHT = 135,50
DATA_PATH = os.path.join(PATH,'data')
#DATA_PATH = os.path.join(PATH,'another_data')
DATA_PATH = os.path.join(PATH,'small_letter_data')
NUM_LETTERS = 5
ITERATIONS = 3
BATCH_SIZE = 64
CAPTCHA_EULER = "01346_c7d87078-3bbf-4c1f-91e0-7189aa3e1a3f.png"
EPOCHS = 5
tim = time.time()
save_dir = os.path.join(PATH, 'saved_models')
model_name = 'keras_cifar10_trained_model_another_modified_db1.h5'
font = '/Library/Fonts/'
FONTS = [font + 'Baskerville.ttc',font+'Chalkboard.ttc',font+'Futura.ttc',font +'Geneva.dfont',font +'GillSans.ttc']


def gen_captcha_dataset(img_dir,num_letters,iteration,width, height,replace = False):
    if replace and os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)
    image  = ImageCaptcha(width = width, height = height,font_sizes=(30,35,40),fonts = FONTS)
    for c in range(iteration):
        print('Generating {}/{}'.format(c,iteration))
        for i in itertools.permutations([str(char) for char in range(10)],num_letters):
            captcha = ''.join(i)
            image.write(captcha,os.path.join(img_dir,'{}_{}.png'.format(captcha,uuid.uuid4())))
            
#gen_captcha_dataset(DATA_PATH,NUM_LETTERS,1,WIDTH,HEIGHT)
#sys.exit(0) 
def load_data(path,num_letters,test_rate = .1):
    print('Loading dataset ...')
    y_train,x_train,y_test,x_test = [],[],[],[]
    
    # r = root, d = directories, f = files
    counter = 0
    for r,d,f in os.walk(path):
        for file in f: 
            
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
                
                img = np.reshape(img,(img.shape[0],img.shape[1],1))
                
                if random() < test_rate :
                    y_test.append(label)
                    x_test.append(img)
                else:
                    y_train.append(label)
                    x_train.append(img)
    print('Dataset size : {}, (train : {}, test : {})'.format(counter, len(y_train),len(y_test)))
    return np.array(x_train).astype('float32')/255,np.array(y_train),np.array(x_test).astype('float32')/255,np.array(y_test)
"""

if not os.path.exists(DATA_PATH):
    print('Generating Dataset')
    gen_captcha_dataset(DATA_PATH, NUM_LETTERS,ITERATIONS, WIDTH, HEIGHT)

x_train, y_train, x_test, y_test = load_data(DATA_PATH,NUM_LETTERS)

s_train = []
s_test = []
for i in range(NUM_LETTERS):
    s_train.append(y_train[:, i, :])
    s_test.append(y_test[:, i, :])
    

print(len(s_train))
"""  
def network():
    input_layer = Input((25, 67, 1))
    x = Conv2D(filters=32, kernel_size=(5, 5), padding='same', activation='relu')(input_layer)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(filters=48, kernel_size=(5, 5), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Conv2D(filters=64, kernel_size=(5, 5), padding='same', activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Dropout(0.3)(x)
    x = Flatten()(x)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.3)(x)
    
    out = [Dense(10, name='digit%d' % i, activation='softmax')(x) for i in range(NUM_LETTERS)]
    model = Model(inputs=input_layer, outputs=out)
    return model

def connection():
    url = 'https://projecteuler.net/sign_in'
    print("Connection to the server")
    driver = webdriver.Safari()
    driver.get(url)
    print("Find image")
    captcha_image_url = driver.find_element_by_id("captcha_image").get_attribute('src')
    print("Downloading image")
    CAPTCHA_EULER = 'captcha_euler3.png'
    open(CAPTCHA_EULER,"wb").write(requests.get(captcha_image_url,allow_redirects = True).content)
    print("Downloading done")
    
    driver.quit()
#connection()

"""
model = network()
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.summary()
print("Start fitting")

history = model.fit(x_train, s_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    verbose=1,
                    validation_data=(x_test, s_test)
                   )

if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)

print('Saved trained model at %s ' % model_path)
"""
"""
hist_train_loss_digit = {i:[] for i in range(5)}
hist_test_loss_digit = {i:[] for i in range(5)}

hist_train_acc_digit = {i:[] for i in range(5)}
hist_test_acc_digit = {i:[] for i in range(5)}

hist_train_loss = []
hist_test_loss = []

hist_train_acc = []
hist_test_acc = []
digit_acc = [[] for _ in range(NUM_LETTERS)]
val_digit_acc = [[] for _ in range(NUM_LETTERS)]
loss = []
val_loss = []


def plot_diagram(digit_acc_now, val_digit_acc_now, loss_now, val_loss_now):
    global digit_acc, val_digit_acc, loss, val_loss
    
    for i in range(NUM_LETTERS):
        digit_acc[i].extend(digit_acc_now[i])
        val_digit_acc[i].extend(val_digit_acc_now[i])
    loss.extend(loss_now)
    val_loss.extend(val_loss_now)
    
    for i in range(NUM_LETTERS):
        s = {0:'First', 1:'Second', 2:'Third', 3:'Fourth', 4:'Fifth'}[i]
        # plt.plot(val_digit_acc[i], label='%s Digit Train' % s)
        plt.plot(digit_acc[i], label='%s Digit Test' % s)

    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()

    for i in range(NUM_LETTERS):
        s = {0:'First', 1:'Second', 2:'Third', 3:'Fourth', 4:'Fifth'}[i]
        plt.plot(val_digit_acc[i], label='%s Digit Train' % s)
        # plt.plot(digit_acc[i], label='%s Digit Test' % s)

    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()



    # Plot training & validation loss values
    plt.plot(val_loss, label='Train')
    plt.plot(loss, label='Test')
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()



if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)
#print("history", history.history)

plot_diagram(
    
    [history.history['digit%d_accuracy' % i] for i in range(NUM_LETTERS)],
    [history.history['val_digit%d_accuracy' % i] for i in range(NUM_LETTERS)],
    history.history['loss'],
    history.history['val_loss'],
)

"""

model_path = os.path.join(save_dir, model_name)
model = load_model(model_path)
"""
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


model.summary()
print("Start fitting")

history = model.fit(x_train, s_train,
                    batch_size=BATCH_SIZE,
                    epochs=EPOCHS,
                    verbose=1,
                    validation_data=(x_test, s_test)
                   )
history.history
model_path = os.path.join(save_dir, model_name[:-3]+'1'+'.h5')
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name[:-3]+'1'+'.h5')
model.save(model_path)
"""
def test_captcha():
    test = 'euler_captcha.jpg'
    test = 'captcha_euler3.png'
    #captcha_path = os.path.join(DATA_PATH,CAPTCHA_EULER)
    captcha_path = os.path.join(PATH,test)
    print(captcha_path)
    img = cv2.imread(captcha_path)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img,(WIDTH//2,HEIGHT//2),interpolation = cv2.INTER_AREA)
    print(img.shape, "sdfsssssssssssssssss")
    img = np.reshape(img,(img.shape[0],img.shape[1],1)).astype('float32')/255
    img = np.expand_dims(img,axis = 0)
    print(img.shape, "sdfsssssssssssssssss")
    #prediction = model.predict(img)
    prediction = model(img,training = False)
    for i in range(NUM_LETTERS):
        print(np.argmax(prediction[i]), prediction[i])
    #print(prediction)
    #prediction = model.predict(np.array([img,]))
    #print(prediction)
    #print("Prediction : ",tensorflow.argmax(prediction,axis = 0))
    
    

#model.summary()

print("Computing time : ", time.time() - tim)

test_captcha()
"""
scores = model.evaluate(x_train, s_train, verbose=1)
print('Train loss:     %f' % np.mean(scores[0:5]))
acc = 1.
for i in range(5):
    acc *= scores[6+i]
print('Train accuracy: %.2f' % (acc * 100.))


scores = model.evaluate(x_test, s_test, verbose=1)
print('Test loss:     %f' % np.mean(scores[0:5]))
acc = 1.
for i in range(5):
    acc *= scores[6+i]
print('Test accuracy: %.2f' % (acc * 100.))
model.predict()

"""
