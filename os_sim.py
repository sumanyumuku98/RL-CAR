import gym
import numpy as np
import pandas as pd
import pickle
import random
from collections import defaultdict

     

class OS():
    """
    Simulate a simple OS cache handler
    """
    def __init__(self, limit, n_pages):
        print(self)
        print(f"Cache limit: {limit}")
        print(f"Total Pages: {n_pages}")
        super(OS, self).__init__()
        self.limit = limit
        self.n_pages = n_pages
        self.P = self.get_P(n_pages)
        if len(self.P) != self.n_pages:
            raise Exception("Size mismatch for P and n_pages")

    @staticmethod
    def get_P(n):
        x = np.random.rand(n)
        x /= x.sum()
        return x 

    def init_pages(self):
        pages = {}
        NT = defaultdict(int)
        starting_pages = random.sample([i for i in range(self.n_pages)], self.limit)
        for i in range(self.limit):
            page_id = starting_pages[i]
            # page_id = i #For now let it be sequential
            lu = 1 #No. of timesteps ago this page was accessed ~LRU
            nt = 1 #No. of times this page was accessed ~LFU
            page = [lu, nt]
            NT[page_id] += 1 # Update NT dict
            pages[page_id] = page
        return pages, NT

    

    def get_id(self):
        return int(np.random.choice(np.arange(self.n_pages), p=self.P))
