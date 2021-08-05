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

if __name__ == "__main__":
    env = gym.make('BipedalWalker-v3')
    #env = gym.make('LunarLanderContinuous-v2')
    #env = gym.make('Pendulum-v0')
    
    
    action_space_high = env.action_space.high[0]
    action_space_low = env.action_space.low[0]
    #print(env.action_space.high,env.action_space.low)
    
    agent = Agent(env.observation_space.shape[0],env.action_space.shape[0],action_space_high,action_space_low)
    #agent.reset()
    sys.exit(0)
    score_history = []
    for episode in range(10):
        observation = env.reset()
        done = False
        score = 0
        while not done :
            
            #action = agent.action(observation)
            action = env.action_space.sample()
            new_observation, reward, done, info = env.step(action)
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
        score_history.append(score)
        print(f"""
                  Épisode :{episode}
                  Score : {score}
                  Avg Score over 10 last episodes : {np.mean(score_history[-10:])}
              """)
    #agent.save_models()
    
            
            
            
        