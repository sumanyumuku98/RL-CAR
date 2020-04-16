import gym
import numpy as np
import pandas as pd
import pickle
from collections import defaultdict

P = [0.3, 0.3, 0.2, 0.15, 0.05]
P =  [0.02592142, 0.12706966, 0.01110356, 0.13603511, 0.15868781,
      0.05753466, 0.11345249, 0.07143165, 0.1702048 , 0.12855884]
P = [0.09048404, 0.08800666, 0.00547254, 0.07488633, 0.03471727,
       0.0627523 , 0.01194143, 0.06526691, 0.08173524, 0.08806114,
       0.02759953, 0.05799802, 0.03055545, 0.06520041, 0.01455946,
       0.0896536 , 0.04006833, 0.0448076 , 0.00798084, 0.0182529 ]      

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
        if len(P) != self.n_pages:
            raise Exception("Size mismatch for P and n_pages")

    def init_pages(self):
        pages = {}
        NT = defaultdict(int)
        for i in range(self.limit):
            #page_id = self.get_id()
            page_id = i #For now let it be sequential
            lu = 0 #No. of timesteps ago this page was accessed ~LRU
            nt = 1 #No. of times this page was accessed ~LFU
            page = [lu, nt]
            NT[page_id] += 1 # Update NT dict
            pages[page_id] = page
        return pages, NT

    

    def get_id(self):
        return int(np.random.choice(np.arange(self.n_pages), p=P))
