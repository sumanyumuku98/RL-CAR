import pickle
from environment import CacheEnv
from collections import defaultdict


def lru_policy(s):
    least_recent = -1
    action = -1
    for key in s.keys():
        cur  = s[key][0]
        if  cur > least_recent:
            action = key
            least_recent = cur
        return action

if __name__ == "__main__":
    l = 20
    trials = 50
    results = defaultdict(list)
    debug = False

    for l in range(10):
        for trial in range(trials):
            env = CacheEnv(eps_len=l)
            s = env.reset()
            done = env.done
            if debug:
                print("Start: ", env.pages)
            while not done:
                a = lru_policy(s)
                s, r, done, observation = env.step(a)
                if debug:
                    print(f">> Request: {env.new_page_id}")
                    print(f"Replace: {a}")
                    print(observation)
                    print(env.pages, f"reward: {r}\n")
            print(f"Total hits: {env.total_hits}")
            results[l].append(env.total_hits)

    with open("results/lru.pkl", "wb") as handle:
        pickle.dump(results, handle)
