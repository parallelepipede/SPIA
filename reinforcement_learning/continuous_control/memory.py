#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 17:01:05 2021

@author: paul
"""

import numpy as np
from collections import deque

class ReplayBuffer:
    def __init__(self,buffer_size,batch_size):
        self.buffer_size = buffer_size
        self.replay_buffer = deque(maxlen=buffer_size)
        self.batch_size = batch_size
        
    def append(self,state,action,reward,next_state):
        self.replay_buffer.append([state,action,np.expand_dims(reward,-1),next_state])
        
    def get_batch(self):
        #np.random.shuffle(self.replay_buffer)
        #self.replay_buffer[:self.batch_size]
        
        return np.random.choice(self.replay_buffer,size=min(len(self.replay_buffer),self.batch_size))
    
    def reset(self):
        self.replay_buffer = deque(maxlen=self.buffer_size)
    
    
    
        
        
        