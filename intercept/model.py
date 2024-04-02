import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import json

c=10 #random exploration rounds

def MoM(X,Y,beta):    
    
    T,m,_ = X.shape
    
    SS = ((Y-X@beta).T @ (Y-X@beta))/T

    sum1 = np.mean(SS)
    sum2 = np.mean(np.diag(SS))

    sigmasq_hat = (sum2 - sum1) * m / (m-1)
    D_hat = sum2 - sigmasq_hat
    
    sigmasq_hat = np.max([0.,sigmasq_hat])
    D_hat = np.max([0.,D_hat])

    return D_hat, sigmasq_hat

class MEUCB:
    def __init__(self, d, K, D, alpha, true_beta, method='True'):
        ## Initialization
        
        self.d = d
        self.K = K
        self.D = D * np.eye(1)
        self.true_beta = true_beta
        self.regrets=[]
        self.beta_errs = []
        self.beta_hat=np.zeros(d)
        self.f = np.zeros(d)
        self.B = np.eye(d)
        self.Binv = np.eye(d)
        self.t = 0
        self.method = method
        ## Hyperparameters
        self.alpha=alpha
        self.settings = {'alpha': self.alpha}

    def select_ac(self, contexts):
        
        self.t+=1
        norms = np.sqrt(np.einsum('Nij,jk,Nik -> Ni', contexts, self.Binv, contexts))
        est = np.sum(contexts @ self.beta_hat + self.alpha * norms, axis=1)

        if self.t>c:
            self.action=np.random.choice(np.where(est == est.max())[0])
        else:
            self.action=np.random.choice(self.K)
        self.contexts=contexts
        self.regrets.append(np.max(np.sum(contexts @ self.true_beta, axis=1)) - np.sum(contexts @ self.true_beta, axis=1)[self.action])
        return(self.action)

    def update(self,reward):
        
        newX=self.contexts[self.action]
        m = len(reward)
        
        if self.method=='True':
            V=np.einsum('ij, jk, lk-> il', np.ones((m,1)), self.D, np.ones((m,1))) + np.eye(m)
            Vinv = np.linalg.inv(V)
        else:
            V=np.eye(m)
            Vinv=np.eye(m)
            
        self.f+=np.einsum('ji, jk, k->i', newX, Vinv, reward)
        self.B+= np.einsum('ji, jk, kl->il', newX, Vinv, newX)
        self.Binv = np.linalg.inv(self.B)
        self.beta_hat=np.dot(self.Binv, self.f)
        self.beta_errs.append(np.sqrt(np.sum((self.beta_hat - self.true_beta)**2)))
        
        
class MEUCB_update:
    def __init__(self, d, K, alpha, true_beta):
        
        self.d = d
        self.K = K
        self.true_beta = true_beta        
        self.regrets=[]
        self.beta_errs = []        
        self.f = np.zeros(d)
        self.B = np.eye(d)
        self.Binv = np.eye(d)
        self.t = 0
        
        #initival value
        self.beta_hat=np.zeros(d)
        self.D_hat = np.eye(1)
        self.sigmasq_hat = 1.0
        
        self.X = []
        self.Y = []
        
        ## Hyperparameters
        self.alpha=alpha
        self.settings = {'alpha': self.alpha}

    def select_ac(self, contexts):
        
        self.t+=1
        norms = np.sqrt(np.einsum('Nij,jk,Nik -> Ni', contexts, self.Binv, contexts)) # k * m array
        est = np.sum(contexts @ self.beta_hat + self.alpha * norms, axis=1) # k array

        if self.t>c:
            self.action=np.random.choice(np.where(est == est.max())[0])
        else:
            self.action=np.random.choice(self.K)
        self.contexts=contexts
        self.regrets.append(np.max(np.sum(contexts @ self.true_beta, axis=1)) - np.sum(contexts @ self.true_beta, axis=1)[self.action])
        return(self.action)

    def update(self,reward):

        newX=self.contexts[self.action]        
        m=len(reward)
        
        self.X.append(newX)
        self.Y.append(reward)       
        X=np.array(self.X)
        Y=np.array(self.Y)
        
        self.D_hat, self.sigmasq_hat = MoM(X,Y,self.beta_hat)
        V = self.D_hat * np.ones((m,m)) + self.sigmasq_hat * np.eye(m)
        Vinv = np.linalg.inv(V)
        
        self.B = np.einsum('Tij,ik,Tkl->jl',X,Vinv,X)
        self.Binv = np.linalg.inv(self.B)
            
        self.f=np.einsum('Tij, ik, Tk->j', X,Vinv,Y)
        
        self.beta_hat=np.dot(self.Binv, self.f)
        self.beta_errs.append(np.sqrt(np.sum((self.beta_hat - self.true_beta)**2)))

def eval_MECUCB1(d, K, m, D=1.0, alpha_set=[0.01, 0.1,1.0], T=1000, M=10, seed=0):

    results = []
    np.random.seed(seed)
    beta = np.random.uniform(-1,1,d)/np.sqrt(d)
    D=D*np.eye(1)

    for alpha in alpha_set:
        
        regrets = np.zeros((M,T))
        errs = np.zeros((M,T))
        
        for repeat in range(M):
            print('ME-CUCB1 Simulation %d, d=%d, K=%d, m=%d, D=%.1f, alpha=%.2f' % (repeat+1, d, K, m, D, alpha))

            model = MEUCB(d=d, K=K, D=D, alpha=alpha, true_beta = beta, method='True')
            
            np.random.seed(seed+repeat)           

            for t in trange(T):
                
                contexts = np.random.randn(K,m,d)
                clip=1./np.maximum(np.ones((K,m)),np.linalg.norm(contexts,axis=2))
                contexts =np.einsum('ijk,ij->ijk',contexts, clip)
                gamma = np.random.multivariate_normal(mean=np.zeros(1), cov=D, size=K)
                eps =  np.random.randn(K,m)
                rewards = contexts @ beta + gamma + eps
                ac = model.select_ac(contexts)
                reward = rewards[ac]

                model.update(reward)
                
            regrets[repeat] = np.cumsum(model.regrets)
            errs[repeat] = model.beta_errs

        results.append({'model':'MECUCB1',
                        'alpha':alpha,
                        'regrets':regrets.tolist(),
                        'errs':errs.tolist()})

    with open('./results/d=%d_K=%d_m=%d_D=%.1f_T=%d_M=%d/MECUCB1.txt' % (d, K, m, D, T, M), 'w') as outfile:
        json.dump(results, outfile)
        
def eval_C2UCB(d, K, m, D=1.0, alpha_set=[0.01, 0.1,1.0], T=1000, M=10, seed=0):

    results = []
    np.random.seed(seed)
    beta = np.random.uniform(-1,1,d)/np.sqrt(d)
    D=D*np.eye(1)

    for alpha in alpha_set:
        
        regrets = np.zeros((M,T))
        errs = np.zeros((M,T))
        
        for repeat in range(M):
            print('C2UCB Simulation %d, d=%d, K=%d, m=%d, D=%.1f, alpha=%.2f' % (repeat+1, d, K, m, D, alpha))

            model = MEUCB(d=d, K=K, D=D, alpha=alpha, true_beta = beta, method='indep')
            
            np.random.seed(seed+repeat)           

            for t in trange(T):
                
                contexts = np.random.randn(K,m,d)
                clip=1./np.maximum(np.ones((K,m)),np.linalg.norm(contexts,axis=2))
                contexts =np.einsum('ijk,ij->ijk',contexts, clip)
                gamma = np.random.multivariate_normal(mean=np.zeros(1), cov=D, size=K)
                eps =  np.random.randn(K,m)
                rewards = contexts @ beta + gamma + eps
                ac = model.select_ac(contexts)
                reward = rewards[ac]
                model.update(reward)
                
            regrets[repeat] = np.cumsum(model.regrets)
            errs[repeat] = model.beta_errs

        results.append({'model':'C2UCB',
                        'alpha':alpha,
                        'regrets':regrets.tolist(),
                        'errs':errs.tolist()})

    with open('./results/d=%d_K=%d_m=%d_D=%.1f_T=%d_M=%d/C2UCB.txt' % (d, K, m, D, T, M), 'w') as outfile:
        json.dump(results, outfile)
        
def eval_MECUCB2(d, K, m, D=1.0, alpha_set=[0.01, 0.1,1.0], T=1000, M=10, seed=0):

    results = []
    np.random.seed(seed)
    beta = np.random.uniform(-1,1,d)/np.sqrt(d)
    D=D*np.eye(1)

    for alpha in alpha_set:
        
        regrets = np.zeros((M,T))
        errs = np.zeros((M,T))
        
        for repeat in range(M):
            print('ME-CUCB2 Simulation %d, d=%d, K=%d, m=%d, D=%.1f, alpha=%.2f' % (repeat+1, d, K, m, D, alpha))
            # call model
            model = MEUCB_update(d=d, K=K, alpha=alpha, true_beta = beta)
            
            np.random.seed(seed+repeat)           

            for t in trange(T):
                
                contexts = np.random.randn(K,m,d)
                clip=1./np.maximum(np.ones((K,m)),np.linalg.norm(contexts,axis=2))
                contexts =np.einsum('ijk,ij->ijk',contexts, clip)
                gamma = np.random.multivariate_normal(mean=np.zeros(1), cov=D, size=K)
                eps = np.random.randn(K,m)
                rewards = contexts @ beta + gamma + eps
                ac = model.select_ac(contexts)
                reward = rewards[ac]

                model.update(reward)
                
            regrets[repeat] = np.cumsum(model.regrets)
            errs[repeat] = model.beta_errs

        results.append({'model':'MECUCB2',
                        'alpha':alpha,
                        'regrets':regrets.tolist(),
                        'errs':errs.tolist()})

    with open('./results/d=%d_K=%d_m=%d_D=%.1f_T=%d_M=%d/MECUCB2.txt' % (d, K, m, D, T, M), 'w') as outfile:
        json.dump(results, outfile)