##command : python3 exp.py --D 1.0
import argparse
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from model import eval_C2UCB, eval_MECUCB1, eval_MECUCB2

parser = argparse.ArgumentParser()

parser.add_argument('--d', default=10, type=int)
parser.add_argument('--K', default=10, type=int)
parser.add_argument('--m', default=10, type=int)
parser.add_argument('--D', default=1.0, type=float)
parser.add_argument('--T', default=10000, type=int)
parser.add_argument('--M', default=10, type=int)

config = parser.parse_args()

d = config.d
K = config.K
m = config.m
D = config.D
T = config.T
M = config.M

if __name__ == '__main__':

    os.makedirs('./results/d=%d_K=%d_m=%d_D=%.1f_T=%d_M=%d' % (d, K, m, D, T, M) ,exist_ok=True)
    alpha_set = [0.0,0.01,0.1,1.0,10.0]
    
    eval_C2UCB(d,K,m,D,alpha_set,T,M,1)
    eval_MECUCB1(d,K,m,D,alpha_set,T,M,1)
    eval_MECUCB2(d,K,m,D,alpha_set,T,M,1)

    results={}
    best={}
    labels = ['$\mathregular{C^2UCB}$', 'ME-CUCB1', 'ME-CUCB2']
    colors = ['C1', 'C0', 'C2']
    
    for i, model in enumerate(['C2UCB', 'MECUCB1', 'MECUCB2']):
        with open(f'./results/d={d}_K={K}_m={m}_D={D:.1f}_T={T}_M={M}/{model}.txt') as infile:
            results[model] = json.load(infile) 
            last_regret = []
            for result in results[model]:
                last_regret.append(np.mean(result['regrets'], axis=0)[-1])
            best[model] = np.argmin(last_regret)
        
        mean = np.mean(results[model][best[model]]['regrets'], axis=0)
        std = np.std(results[model][best[model]]['regrets'], axis=0)
       
        plt.plot(np.arange(1,T+1),mean,label=labels[i], color=colors[i])
        plt.fill_between(np.arange(1,T+1), mean - 1.96/np.sqrt(M) * std, mean + 1.96/np.sqrt(M) * std, alpha=0.1, color=colors[i])

    plt.legend()
    plt.title(f'Cumulative Regrets: d={d}, K={K}, m={m}, D={D:.1f}')
    plt.xlabel('Number of rounds')
    plt.ylabel('Cumulative Regrets')
    plt.savefig(f'./results/d={d}_K={K}_m={m}_D={D:.1f}_T={T}_M={M}/regrets_d={d}_K={K}_m={m}_D={D:.1f}_T={T}_M={M}.pdf', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()
    
    for i, model in enumerate(['C2UCB', 'MECUCB1', 'MECUCB2']):
        with open(f'./results/d={d}_K={K}_m={m}_D={D:.1f}_T={T}_M={M}/{model}.txt') as infile:
            results[model] = json.load(infile) 
            last_regret = []
            for result in results[model]:
                last_regret.append(np.mean(result['regrets'], axis=0)[-1])
            best[model] = np.argmin(last_regret)
            
        mean = np.mean(results[model][best[model]]['errs'], axis=0)
        std = np.std(results[model][best[model]]['errs'], axis=0)
       
        plt.plot(np.arange(1,T+1),mean,label=labels[i], color=colors[i])
        plt.fill_between(np.arange(1,T+1), mean - 1.96/np.sqrt(M) * std, mean + 1.96/np.sqrt(M) * std, alpha=0.1, color=colors[i])

    plt.legend()
    plt.title(f'Estimation Errors: d={d}, K={K}, m={m}, D={D:.1f}')
    plt.xlabel('Number of rounds')
    plt.ylabel(r'$\|| \beta - \widehat{\beta}\||_{2}$')
    plt.ylim((0,0.5))
    plt.savefig(f'./results/d={d}_K={K}_m={m}_D={D:.1f}_T={T}_M={M}/errs_d={d}_K={K}_m={m}_D={D:.1f}_T={T}_M={M}.pdf', bbox_inches='tight', pad_inches=0, dpi=300)
    plt.close()