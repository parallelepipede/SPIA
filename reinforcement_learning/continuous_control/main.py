#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Aug  2 00:03:39 2021

@author: paul
"""

# continuous paper : https://arxiv.org/pdf/1509.02971.pdf

import gym
import os
import numpy as np
import tensorflow as tf

# Need a replay buffer class
# Need a class for target Q networks (function of s and a)
# Batch norm
# policy : deterministic : how to handle explore exploit ?
# Deterministic action means outputs the actual action instead of a probability of action
# Will need a way to bound the actions due to the env limits.
# We have two actor and two critic  networks, a target for each.
# Updates are soft. according to theta_prime = tau*theta + (1-tau)*theta_prime. tau << 1
# the target actor is just the evaluation actor plus some noice process -> will need a class for the noice.

 
class OrnsteinUhlenbeckActionNoise(object):
    # Comes from Open AI library : https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
    def __init__(self,mu,sigma=.2,theta=.15,dt=1e-2,x0=None):
        self.theta = theta
        self.mu = mu
        self.dt = dt
        self.x0 = x0
        self.sigma = sigma
        self.reset()
    
    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
        
    def __repr__(self):
        return f'OrnsteinUhlenbeck(mu={self.mu},sigma={self.sigma})'
    
    def __call__(self):
        # noise = OrnsteinUhlenbeckActionNoise()
        # ourNoise = noise() -> __call__ is called
        x = self.x_prev  + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size = self.mu.shape)
        self.x_prev = x
        return x
    

class ReplayBuffer(object):
    def __init__(self,max_size,input_shape,n_actions):
        
        """ 
        need to store the state action reward and new state tuples
        want to facilitate the use of the done flags so need for an extra parameter
        related to Bellman equation.
        At the end of the episode, the agent receives no further rewards, 
        so the expected feature reward, the discounted  feature reward is 0.
        """
        self.mem_size = max_size
        self.mem_cntr = 0
        self.state_memory = np.zeros((self.mem_size,*input_shape))
        self.new_state_memory = np.zeros((self.mem_size,*input_shape))
        self.action_memory = np.zeros((self.mem_size,n_actions))
        self.input_shape = input_shape
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size, dtype=np.float32)
        
    def store_transition(self,state,action,reward,state_,done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.action_memory[index] = action
        self.terminal_memory[index] = 1 - done
        self.mem_cntr += 1
    
    def sample_buffer(self,batch_size):
        # fill this line
        max_mem = min(self.mem_cntr,self.mem_size)
        batch = np.random.choice(max_mem,batch_size)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        new_states = self.new_state_memory[batch]
        rewards = self.reward_memory[batch]
        terminals = self.terminal_memory[batch]
        
        return states,actions,rewards,new_states,terminals
        

class Actor(object):
    def __init__(self,lr,n_actions,name,input_dims, sess,
                 fc1_dims,fc2_dims,action_bound, batch_size = 64, chkpt_dir="tmp/ddpg"):
        self.lr = lr
        self.n_actions = n_actions
        self.name = name
        self.fc1_dims = fc1_dims
        self.fc2_dims = fc2_dims
        self.sess = sess
        self.batch_size = batch_size
        self.action_bound = action_bound
        self.chkpt_dir = chkpt_dir
        self.build_network()
        self.params = tf.trainable_variables(scope = self.name)
        self.saver = tf.train.Saver()
        self.checkpoint_file = os.path.join(self.chkpt_dir,self.name+'_ddpg.ckpt')
        
        self.unormalized_actor_gradients = tf.gradients(self.mu, self.params, -self.action_gradient)
        self.actor_gradients = list(map(lambda x : tf.math.divide(x,self.batch_size),self.unormalized_actor_gradients))    
        self.optimize = tf.train.
        
        
        
        
        
        
        
        
        
        
        
        
        
    
    
    