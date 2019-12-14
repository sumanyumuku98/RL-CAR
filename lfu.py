import pickle
from environment import CacheEnv
from collections import defaultdict


def lfu_policy(s):
    least_frequent = 100000000
    action = -1
    for key in s.keys():
        cur = s[key][1]
        if cur < least_frequent:
            action = key
            least_frequent = cur
        return action

if __name__ == "__main__":
    l = 20
    trials = 50
    results = defaultdict(list)
    debug = False

    for l in range(1, 10):
        for trial in range(trials):
            env = CacheEnv(eps_len=l)
            s = env.reset()
            done = env.done
            if debug:
                print("Start: ", env.pages)
            while not done:
                a = lfu_policy(s)
                s, r, done, observation = env.step(a)
                if debug:
                    print(f">> Request: {env.new_page_id}")
                    print(f"Replace: {a}")
                    print(observation)
                    print(env.pages, f"reward: {r}\n")
            print(f"Total hits: {env.total_hits}")
            percentage = 100 * env.total_hits / l
            results[l].append(percentage)

    with open("results/lfu.pkl", "wb") as handle:
        pickle.dump(results, handle)
