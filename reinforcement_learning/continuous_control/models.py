#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 00:03:39 2021

@author: paul
"""

# continuous paper : https://arxiv.org/pdf/1509.02971.pdf

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import optimizers,Input, Model
from tensorflow.keras.layers import Dense, BatchNormalization, Conv2D,Add
from memory import ReplayBuffer

# Need a replay buffer class
# Need a class for target Q networks (function of s and a)
# Batch norm
# policy : deterministic : how to handle explore exploit ?
# Deterministic action means outputs the actual action instead of a probability of action
# Will need a way to bound the actions due to the env limits.
# We have two actor and two critic  networks, a target for each.
# Updates are soft. according to theta_prime = tau*theta + (1-tau)*theta_prime. tau << 1
# the target actor is just the evaluation actor plus some noice process -> will need a class for the noice.

"""

class Actor(object):
    def __init__(self,lr,n_actions,name,input_dims,
                 fc1_dims,fc2_dims,action_bound, batch_size = 64, chkpt_dir="tmp/ddpg"):
        self.lr = lr
        self.n_actions = n_actions
        self.name = name
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.batch_size = batch_size
        self.action_bound = action_bound
        self.chkpt_dir = chkpt_dir
        self.build_network()
        self.params = tf.Variable(import_scope = self.name)
        self.checkpoint_file = os.path.join(self.chkpt_dir,self.name+'_ddpg.ckpt')
        
        self.unormalized_actor_gradients = tf.gradients(self.mu, self.params, - self.action_gradient)
        self.actor_gradients = list(map(lambda x : tf.math.divide(x,self.batch_size),self.unormalized_actor_gradients))    
        #self.optimize = optimizers.Adam(self.lr).
        self.optimize = optimizers.Adam(self.lr).apply_gradients(zip(self.actor_gradients,self.params))
        
    def build_network(self):
         with tf.variable_scope(self.name, reuse=tf.AUTO_REUSE):
             f1 = 1./np.sqrt(self.fc1_dims)
             x = Input(shape=(None,*self.input_dims),batch_size =self.batch_size, name ="inputs",dtype=tf.float32)
             x = tf.layers.dense(x,self.n_actions,kernel_initializer=tf.random_uniform_initializer(minval=-f1,maxval=f1)) #biais_initializer(t)
             
             #x = tf.nn.relu(x)
             x = tf.tanh(x)
             
             f2 = 1./np.sqrt(self.fc2_dims)
             
    def save_model(self):
        return 
"""

def update_target_network(model_target,model_ref,rho):
    model_target.set_weights([rho * ref_weight + (1 - rho) * target_weight 
                              for (target_weight,ref_weight) in zip(model_target.get_weights(),model_ref.get_weights())])
    
# Read Experiments and algorithm section for all details

class Actor:
    
    def __init__(self,batch_size=64):
        self.batch_size = batch_size
        self.kernel_initializer=tf.random_uniform_initializer(minval=-3e-3,maxval=3e-3)
        self.initializer = tf.keras.initializers.VarianceScaling(scale=0.1, mode='fan_in', distribution='uniform')
        
    def build_network(self,n_states,n_actions,action_high):
        inputs = Input(shape = (n_states,),batch_size = self.batch_size)
        x = Dense(400,activation = tf.nn.relu,kernel_initializer = self.initializer, name = "actor_f1")(inputs)  #kernel_initializer = self.initializer
        x = Dense(300,activation = tf.nn.relu,kernel_initializer = self.initializer, name = "actor_f2")(x) #kernel_initializer = self.initializer,
        outputs = Dense(n_actions,activation = "tanh",kernel_initializer=self.kernel_initializer)(x) * action_high
        
        self.model = Model(inputs,outputs)
        return self.model
        
class Critic:
    def __init__(self,batch_size=64):
        self.batch_size = batch_size
        self.kernel_initializer=tf.random_uniform_initializer(minval=-3e-4,maxval=3e-4)
        self.initializer = tf.keras.initializers.VarianceScaling(scale=0.1, mode='fan_in', distribution='uniform')
        
    def build_network(self,n_states,n_actions,action_high):
        #include states
        state_inputs = Input(shape = (n_states,),batch_size = self.batch_size)
        x = Dense(400,activation = tf.nn.relu,kernel_initializer = self.initializer, name = "critic_f1")(state_inputs)
        x = BatchNormalization()(x)
        state_outputs = Dense(300,activation = tf.nn.relu,kernel_initializer = self.initializer, name = "critic_f2")(x)
        
        #include actions
        action_inputs = Input(shape = (n_actions,), batch_size = self.batch_size)
        action_outputs = Dense(300, activation=tf.nn.relu,kernel_initializer = self.initializer,name="critic_f3")(action_inputs / action_high)
        
        #add both states and actions
        added_layer = Add()([state_outputs, action_outputs])
        added_layer = BatchNormalization()(added_layer)
        
        x = tf.keras.layers.Dense(150, activation=tf.nn.relu,kernel_initializer = self.initializer,name="critic_f4")(added_layer)
        x = tf.keras.layers.BatchNormalization()(x)
        outputs = tf.keras.layers.Dense(1, kernel_initializer=self.kerner_initializer,name="critic_f5")(x)
        
        self.model = tf.keras.Model([state_inputs, action_inputs], outputs)
        return self.model
        
        
        
class Agent(object):
    def __init__(self, n_states, n_actions, action_high, action_low, gamma=.99, rho=.001,std_dev=.2):
        self.actor = Actor().build_network(n_states,n_actions,action_high)
        self.critic = Critic().build_network(n_states,n_actions,action_high)
        self.replay_buffer = ReplayBuffer(buffer_size = 1e6,batch_size = 64)
        
    def action(self,state):
        self.actor
        
    def remember(self,state,reward,action,next_state):
        self.replay_buffer.append(state,reward,action,next_state)
        
        
        
        
        
        
        
    
    
    