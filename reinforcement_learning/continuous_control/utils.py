#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 15:21:44 2021

@author: paul
"""

import datetime
import os
import tensorflow as tf


class Tensorboard:
    #https://www.tensorflow.org/tensorboard/migrate
    def __init__(self,log_dir):
        current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        if not os.path.exists(log_dir):
            os.mkdir(log_dir)
        log_dir = os.path.join(log_dir,current_time)
        self.writer = tf.summary.create_file_writer(log_dir,name="writer")
        
    
    def __call__(self,epoch,action,reward,loss_c,loss_a):
        with self.writer.as_default():
            tf.summary.scalar("Action",action,step=epoch)
            tf.summary.scalar("Reward",reward,step=epoch)
            tf.summary.scalar("loss_c",loss_c,step=epoch)
            tf.summary.scalar("loss_a",loss_a,step=epoch)
        
        

            
            
        