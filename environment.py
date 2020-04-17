import gym
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict
from os_sim import OS

LIMIT = 3
N_PAGES = 5
EPS_LEN = 3
POS_REW = 1
NEG_REW = -1
HEAVY_NEG_R = -10

class CacheEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self, limit=LIMIT, n_pages=N_PAGES, eps_len=EPS_LEN):
        super(CacheEnv, self).__init__()
        self.limit = limit
        self.n_pages = n_pages
        self.eps_len = eps_len
        self.os = OS(limit, n_pages)
        self.pages, self.NT = self.os.init_pages()
        self.total_hits = 0
        self.timestep = 0 #counter; if this reaches eps_len, return done=True
        self.done = False
        self.new_page_id = -1
        self.action_space_n = limit

    def step(self, action, test=False):
        """
        First OS will send a page id (randomly from distribution P)
        based on the action choose to evict a page
        allocate this page id cache inplace of the 'action' id
        """
        self.timestep += 1
        if self.timestep >= self.eps_len:
            self.done = True #Episode reached its end
        new_page_id = self.os.get_id() #This is page requested by the OS
        self.new_page_id = new_page_id #Store for debugging
        reward, hit = self.allocate_cache(action, new_page_id)
        if hit:
            observation = f"This was a hit, OS asked for: {new_page_id}"
            self.total_hits += 1
        else:
            observation = f"This was not a hit, OS asked for: {new_page_id}"
        # return self.pages, reward, self.done, observation
        return self.nn_state(), reward, self.done, observation

    def reset(self):
        self.timestep = 0
        self.pages, self.NT = self.os.init_pages()
        self.total_hits = 0 #Intuitive
        self.done = False
        # return self.pages
        return self.nn_state()

    def render(self, mode='human'):
        pass

    def close(self):
        pass

    def if_allocated(self, id):
        """
        returns true if 'id' is allocated a cache currently
        """
        if id in self.pages.keys():
            return True
        return False

    def nn_state(self):
        """returns state in numpy format for neural net inpu"""
        state = []
        for k in self.pages:
            vals = self.pages[k]
            state.append(vals[0]) #Flatten
            state.append(vals[1]) #Flatten
        return np.array(state)        

    def allocate_cache(self, action, id):
        """
        remove page at 'action'
        add page 'id'
        """
        action = int(action)
        id = int(id)
        hit = False #Page hit or not?
        self.NT[id] += 1
        # For all the pages except id, increament their lu counter
        for page_id in self.pages.keys():
            if page_id == id:
                continue
            else:
                self.pages[page_id][0] += 1
                # new_page = self.pages[page_id]
                # new_page = [new_page[0]+1, new_page[1]]
                # self.pages[page_id] = new_page

        # if action not in self.pages.keys():
        #     #Agent asked to remove a page that wasn't even allocated
        #     return HEAVY_NEG_R, hit

        if self.if_allocated(id):
            hit = True #HIT!
            page = self.pages[id]
            page[0] = 0
            page[1] += 1
            self.pages[id] = page
            reward = POS_REW #pos reward for hit
        else:
            key = list(self.pages.keys())[action]
            self.pages.pop(key) #Remove page 'action'
            self.pages[id] = [0, self.NT[id]] #Add page 'id'
            reward = NEG_REW #neg reward for no hit

        return reward, hit

if __name__ == "__main__":
    env = CacheEnv()
    env.reset()
