#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 13 18:48:18 2020

@author: shashank
"""
from gym_torcs import TorcsEnv
import numpy as np
import matplotlib.pyplot as plt
import torch

device = torch.device("cuda" if True else "cpu")
from lib.Model import ActorCritic

#### Generate a Torcs environment
# enable vision input, the action is steering only (1 dim continuous action)
env1 = TorcsEnv(vision=True, throttle=False)
# env2 = TorcsEnv(vision=True, throttle=False)

# without vision input, the action is steering and throttle (2 dim continuous action)
# env = TorcsEnv(vision=False, throttle=True)
model = ActorCritic(32, 2, 200).to(device)

ob1 = env1.reset(relaunch=True)  # with torcs relaunch (avoid memory leak bug in torcs)
# ob = env2.reset(relaunch=False)  # with torcs relaunch (avoid memory leak bug in torcs)
# ob = env.reset()  # without torcs relaunch

# Generate an agent
# from sample_agent import Agent
# agent = Agent(1)  # steering only
# action = agent.act(ob, reward, done, vision=True)

# single step
# while True:
for i in range(300000):
    ob1, reward, done, _ = env1.step(np.random.random((1,)))
# ob2, reward, done, _ = env2.step(np.random.random((1,)))
# plt.imshow(ob1)
# plt.show
state = torch.FloatTensor([ob1]).to(device)
dist, value = model(state)

# shut down torcs
env1.end()
# env2.end()