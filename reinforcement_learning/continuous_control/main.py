#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  5 15:21:44 2021

@author: paul
"""

import gym
gym.logger.set_level(40) # Block warning
import numpy as np
from models import Agent
import sys
from utils import Tensorboard
import tensorflow as tf

if __name__ == "__main__":
    env = gym.make('BipedalWalker-v3')
    #env = gym.make('LunarLanderContinuous-v2')
    #env = gym.make('Pendulum-v0')
    
    
    action_space_high = env.action_space.high[0]
    action_space_low = env.action_space.low[0]
    #print(env.action_space.high,env.action_space.low)
    
    agent = Agent(env.observation_space.shape[0],env.action_space.shape[0],action_space_high,action_space_low)
    #agent.reset()
    tensorboard = Tensorboard("models")
    score_history = []
    acc_reward = tf.keras.metrics.Sum('reward', dtype=tf.float32)
    actions_squared = tf.keras.metrics.Mean('actions', dtype=tf.float32)
    #loss_c = tf.keras.metrics.Mean('loss_c', dtype=tf.float32)
    loss_c = tf.keras.metrics.MeanSquaredError('loss_c', dtype=tf.float32)
    #loss_a = tf.keras.metrics.Mean('loss_a', dtype=tf.float32)
    loss_a = tf.keras.metrics.MeanSquaredError('loss_a', dtype=tf.float32)
    for episode in range(10):
        observation = env.reset()
        agent.noise.reset()
        done = False
        score = 0
        for _ in range(100):
            env.render()
            action = agent.action(observation)
            #action = env.action_space.sample()
            new_observation, reward, done, info = env.step(action)
            agent.remember(new_observation,reward,action,observation)
            agent.learn()
            print("bein",
                  new_observation)
            print(reward)
            print(done)
            print(info)
            print(observation, " observation")
            sys.exit(0)
            #agent.remember(observation,action,new_observation,done)
            #agent.learn()
            
            score += reward
            observation = new_observation
            env.render()
            continue
            print(f"""Observation : {new_observation}
                      Reward : {reward}
                      done : {done}
                      info : {info}
                """)
            if done : 
                break
        score_history.append(score)
        print(f"""
                  Épisode :{episode}
                  Score : {score}
                  Avg Score over 10 last episodes : {np.mean(score_history[-10:])}
              """)
        observation = env.reset()
        acc_reward.reset_states()
        actions_squared.reset_states()
        loss_c.reset_states()
        loss_a.reset_states()
    #agent.save_models()
    
            
            
            
        