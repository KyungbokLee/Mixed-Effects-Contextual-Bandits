import argparse
import time
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from model_avg import MoM, REUCB,REUCB_avg, REUCB_update
from tqdm import trange

parser = argparse.ArgumentParser()

parser.add_argument('--K', default=10, type=int)
parser.add_argument('--m', default=5, type=int)
parser.add_argument('--T', default=1000, type=int)
parser.add_argument('--M', default=5, type=int)
parser.add_argument('--c', default=100, type=int)
parser.add_argument('--seed', default=0, type=int)

config = parser.parse_args()

K = config.K
m = config.m
T = config.T
M = config.M
c = config.c
seed = config.seed

d=10

if __name__ == '__main__':
    
    t=time.time()
    
    print('Loading Dataset...')
    data = pd.read_pickle('test_10m.pickle')
    print(f'Dataset Loaded in {time.time()-t:.1f}s.')
    v=data['user'].value_counts()
    users = data[data['user'].isin(v.index[v.gt(K*m)])]['user'].unique()
    
    alpha_set = [0.001, 0.01, 0.1, 1.0, 10.0]
    
    results1 = []
    results2 = []   
    results3 = []  

#indep model
    for alpha in alpha_set:
        np.random.seed(seed)
        
        regrets1 = np.zeros((M,T))
        regrets2 = np.zeros((M,T))
        regrets3 = np.zeros((M,T))
        
        for repeat in range(M):
            
            print('Simulation %d, d=%d, K=%d, c=%d, m=%d, alpha=%.2f' % (repeat+1, d, K, c, m, alpha))
            # call model
            model1 = REUCB(K=K, c=c, d=d, alpha=alpha)
            model2 = REUCB_update(K=K, c=c, d=d, alpha=alpha)
            model3 = REUCB_avg(K=K, c=c, d=d, alpha=alpha)
            
            np.random.seed(seed+repeat)           

            for t in trange(T):           
                user = np.random.choice(users)
                user_data = data.query('user==@user')[:K*m]

                rewards = np.array(user_data['rating']).reshape((K,m))                
                contexts = np.array(list(user_data['context'])).reshape((K,m,-1))
                
                ac1 = model1.select_ac(contexts)
                ac2 = model2.select_ac(contexts)
                ac3 = model3.select_ac(contexts)
                
                reward1 = rewards[ac1]
                reward2 = rewards[ac2]
                reward3 = rewards[ac3]

                max_reward = np.max(np.mean(rewards,axis=1))

                model1.update(reward1)
                model2.update(reward2)
                model3.update(reward3)
                
                regrets1[repeat,t] = max_reward - np.mean(reward1)
                regrets2[repeat,t] = max_reward - np.mean(reward2)
                regrets3[repeat,t] = max_reward - np.mean(reward3)

        results1.append(regrets1)
        results2.append(regrets2)
        results3.append(regrets3)
        
    best1 = 0
    best2 = 0
    best3 = 0
    best1_sum = np.inf
    best2_sum = np.inf
    best3_sum = np.inf
    
    for i in range(len(alpha_set)):
        if np.sum(results1[i])<best1_sum:
            best1 = i
            best1_sum = np.sum(results1[i])
            
        if np.sum(results2[i])<best2_sum:
            best2 = i
            best2_sum = np.sum(results2[i])

        if np.sum(results3[i])<best3_sum:
            best3 = i
            best3_sum = np.sum(results3[i])            
    
    mean1 = np.cumsum(np.mean(results1[best1], axis=0))
    mean2 = np.cumsum(np.mean(results2[best2], axis=0))
    mean3 = np.cumsum(np.mean(results3[best3], axis=0))
    
    std1 = np.std(np.cumsum(results1[best1], axis=1),axis=0)
    std2 = np.std(np.cumsum(results2[best2], axis=1),axis=0)
    std3 = np.std(np.cumsum(results3[best3], axis=1),axis=0)

    #print(f'ME-CUCB2: half mean {mean2[-500]}, std {std2[-500]}')
    #print(f'LinUCB: half mean {mean3[-500]}, std {std3[-500]}')

    print(f'ME-CUCB2: mean {mean2[-1]}, std {std2[-1]}')
    print(f'LinUCB: mean {mean3[-1]}, std {std3[-1]}')
  
    os.makedirs('./results_avg/random_K=%d_d=%d_m=%d_c=%d_T=%d_M=%d' % (K, d, m, c, T, M) ,exist_ok=True)
    
    plt.figure(figsize=(6.4,3.0))
    
    plt.plot(mean1,label='$\mathregular{C^2UCB}$', color='C1')
    plt.fill_between(range(T), mean1 - 1.96/np.sqrt(M) * std1, mean1 + 1.96/np.sqrt(M) * std1, alpha=0.2, color='C1')
    plt.plot(mean2,label='ME-CUCB2', color='C2')
    plt.fill_between(range(T), mean2 - 1.96/np.sqrt(M) * std2, mean2 + 1.96/np.sqrt(M) * std2, alpha=0.2, color='C2')
    plt.plot(mean3,label='LinUCB', color='C3')
    plt.fill_between(range(T), mean3 - 1.96/np.sqrt(M) * std3, mean3 + 1.96/np.sqrt(M) * std3, alpha=0.2, color='C3')
    
    plt.xlabel('Number of Rounds')
    plt.ylabel('Cumulative Regret')
    
    plt.title('Cumulative Regrets : MovieLens Dataset')

    plt.legend()
    
    plt.savefig(f'./results_avg/random_K={K}_d={d}_m={m}_c={c}_T={T}_M={M}/regrets_movielens.pdf', bbox_inches='tight', pad_inches=0, dpi=300)