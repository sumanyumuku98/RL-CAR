import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from collections import namedtuple
import math
import random


class DQN(object):
    def __init__(self):
        self.target_net = Net().to(torch.device('cuda'))
        self.policy_new = Net().to(torch.device('cuda'))
        self.learn_step_counter = 0 #For target net updation
        self.observe_counter = 0 #For storage
        # self.memory = np.zeros((MEMORY_CAPACITY, N_STATES * 2 + 2))
        self.memory = ReplayMemory(MEMORY_CAPACITY)
        self.optimizer = torch.optim.Adam(self.policy_new.parameters(), lr=LR)
        self.loss_func = nn.SmoothL1Loss()
        self.PATH = 'model/'
        self.epsilon = EPS_START
        self.step = 0

    def choose_action(self, x, test=False):
        self.step += 1
        print(f"Epsilon: {self.epsilon}")
        if random.random() <= self.epsilon:
          action = np.random.randint(0, N_ACTIONS)
          print(f"[RANDOM {action}]")
        else:
          with torch.no_grad():
              x = torch.unsqueeze(torch.FloatTensor(x).to(torch.device('cuda')), 0).permute(0, 3, 1, 2)
              action = int(self.policy_new.forward(x).max(1)[1].view(1, 1))
              print(f"[NET {action}]")
        if self.epsilon > EPS_END and self.step > E_OBSERVE_STEPS:
          old_e = self.epsilon
          interval = EPS_START - EPS_END
          self.epsilon -= interval / float(E_EXPLORE_STEPS)
        return action

    def store_transition(self, s, a, r, s_):
        self.memory.push(s, a, r, s_)

    def learn(self):
        if self.observe_counter < E_OBSERVE_STEPS:
            self.observe_counter += 1
            return
        if self.learn_step_counter % TARGET_REPLACE_ITER == 0:
            print("Traget updated!")
            self.target_net.load_state_dict(self.policy_new.state_dict())
        self.learn_step_counter += 1

        batch = Transition(*zip(*self.memory.sample(BATCH_SIZE)))
        #https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
        # ^For explanation of above

        b_s = torch.Tensor(batch.state).permute(0, 3, 1, 2).cuda()
        b_s_ = torch.Tensor(batch.next_state).permute(0, 3, 1, 2).cuda()
        b_a = torch.LongTensor(batch.action)[..., None].cuda()
        b_r = torch.FloatTensor(batch.reward)[..., None].cuda()
        q_eval = self.policy_new(b_s).gather(1, b_a)
        q_next = self.target_net(b_s_).detach()
        q_target = b_r + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.policy_new.parameters():
            # param.grad.data.clamp_(-1, 1) #Gradient clipping
        self.optimizer.step()

    def save_model(self, name):
        eval_name = name + '_policy_new.m'
        train_name = name + '_train_net.m'
        torch.save(self.policy_new.state_dict(), self.PATH  + eval_name)
        torch.save(self.target_net.state_dict(), self.PATH  + train_name)

    def load_model(self, name):
        eval_name = name + '_policy_new.m'
        train_name = name + '_train_net.m'
        self.policy_new.load_state_dict(torch.load(self.PATH + eval_name))
        self.target_net.load_state_dict(torch.load(self.PATH  + train_name))
