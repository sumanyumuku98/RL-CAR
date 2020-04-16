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
    def __init__(self, limit=LIMIT, n_pages=N_PAGES, eps_len=EPS_LEN, human=False):
        super(CacheEnv, self).__init__()
        self.limit = limit
        self.n_pages = n_pages
        self.eps_len = eps_len
        self.os = OS(limit, n_pages)
        self.pages, self.NT = self.os.init_pages()
        self.timestep = 0 #counter; if this reaches eps_len, return done=True
        self.done = False
        self.new_page_id = -1
        self.action_space_n = limit
        self.human = human

    def step(self, action, test=False):
        """
        OS just asked for a page not in memory (stored in self.new_page_id).
        Replace page at `action` index to make space for the page. 
        Then keep asking OS for more pages until a miss occurs. 
        For ever hit meanwhile, increase positive reward by 1.
        """
        self.timestep += 1
        if self.timestep >= self.eps_len:
            self.done = True #Episode reached its end
        
        
        _, hit = self.allocate_cache(self.new_page_id, action) #we took action to make space for this page
        assert(not hit), "Something weird happened just now..."

        reward = 0
        nhits = 0
        hit = True
        while(hit): #until page miss occurs
            new_page_id = self.os.get_id() #This is page requested by the OS
            self.new_page_id = new_page_id #Store for debugging
            r, hit = self.allocate_cache(new_page_id, ignore_miss=True)
            reward += r
            nhits += 1

        observation = f"There were {nhits} hits."
        return self.nn_state(), reward, self.done, observation

    def reset(self):
        self.timestep = 0
        self.pages, self.NT = self.os.init_pages() #self.NT keeps record of number of times a page was accessed. This info might be lost after page is removed so its saved here.
        self.done = False
        self.new_page_id = self.page_not_in_memory() #this will cause a miss and make agent choose an action for step.
        # return self.pages
        return self.nn_state()

    def page_not_in_memory(self):
        current_pages = set(self.pages.keys())
        all_pages = set([i for i in range(self.n_pages)])    
        left_pages = list(all_pages-current_pages)
        return np.random.choice(left_pages)



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

    def toggle_human(self):
        self.human = not self.human

    def nn_state(self):
        """returns state in numpy format for neural net inpu"""
        if self.human:
            return self.pages
        state = []
        for k in self.pages:
            vals = self.pages[k]
            state.append(vals[0]) #Flatten
            state.append(vals[1]) #Flatten
        return np.array(state)        

    def allocate_cache(self, id, action=None, ignore_miss=False):
        """
        remove page at 'action'
        add page 'id'
        use `ignore_miss` in cases when you want to return `false` for hit and not replace cache.
        """
        id = int(id)
        hit = False #Page hit or not?
        self.NT[id] += 1
        # For all the pages except id, increament their lu counter
        for page_id in self.pages.keys():
            if page_id == id:
                continue
            else:
                self.pages[page_id][0] += 1

        if self.if_allocated(id):
            hit = True #HIT!
            page = self.pages[id]
            page[0] = 0
            page[1] += 1
            self.pages[id] = page
            reward = POS_REW #pos reward for hit
        elif ignore_miss:
            return NEG_REW, hit    
        else:
            action = int(action)
            key = list(self.pages.keys())[action]
            self.pages.pop(key) #Remove page 'action'
            self.pages[id] = [0, self.NT[id]] #Add page 'id' (load how frequently this page is used from a global counter: NT)
            reward = NEG_REW #neg reward for no hit

        return reward, hit

if __name__ == "__main__":
    env = CacheEnv()
    env.reset()
