import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import json

def Estep(X,Y,Z,beta,D,sigmasq):
    m = Z.shape[1]
    V = np.einsum('Tij, jk, Tlk -> Til',Z, D, Z) + sigmasq * np.eye(m)
    Vinv = np.linalg.inv(V)
    e = Y - np.einsum('Tij, j->Ti', X, beta)
    Eb = np.einsum('ij, Tkj, Tkl, Tl -> Ti', D, Z, Vinv, e)
    Varb = D - np.einsum('ij, Tkj, Tkl, Tlm, mn -> Tin', D, Z, Vinv, Z, D)
    
    return Eb, Varb

def Mstep(X,Y,Z,beta,D,sigmasq,Eb,Varb):
    T = len(X)
    m = Z.shape[1]
    B = np.einsum('Tij,Tik->jk',X,X)
    Binv = np.linalg.inv(B)
    e1 = Y - np.einsum('Tij, Tj->Ti', Z, Eb)
    beta_new = np.einsum('ij,Tkj,Tk->i', Binv, X, e1)
    
    D_new = (np.sum(Varb, axis=0) + np.einsum('Ti,Tj->ij', Eb, Eb))/T
    
    e2 = Y - np.einsum('Tij, j->Ti', X, beta) - np.einsum('Tij, Tj->Ti', Z, Eb)
    sigmasq_new = (np.einsum('Ti,Ti->',e2,e2) + np.einsum('Tij,Tjk,Tik->',Z,Varb,Z))/(m*T)
    
    return beta_new, D_new, sigmasq_new

def EM(X,Y,Z,beta_init,D_init,sigmasq_init,loop=5):    
    
    beta_new, D_new, sigmasq_new = beta_init, D_init, sigmasq_init
    
    for i in range(loop):
        Eb, Varb = Estep(X,Y,Z,beta_new,D_new,sigmasq_new)
        beta_new, D_new, sigmasq_new = Mstep(X,Y,Z,beta_new,D_new,sigmasq_new,Eb,Varb)
 
    return beta_new, D_new, sigmasq_new
sigma = 0.25

class MEUCB:
    def __init__(self, d, K, D, alpha, true_beta, method='True', c=20):
        ## Initialization
        
        self.d = d
        self.K = K
        self.D = D * np.eye(d)
        self.true_beta = true_beta
        self.regrets=[]
        self.beta_errs = []
        self.beta_hat=np.zeros(d)
        self.f = np.zeros(d)
        self.B = np.eye(d)
        self.Binv = np.eye(d)
        self.t = 0
        self.c = c
        self.method = method
        
        ## Hyperparameters
        self.alpha=alpha
        self.settings = {'alpha': self.alpha}

    def select_ac(self, contexts):
        
        self.t+=1
        norms = np.sqrt(np.einsum('Nij,jk,Nik -> Ni', contexts, self.Binv, contexts))
        est = np.sum(contexts @ self.beta_hat + self.alpha * norms, axis=1)
        
        if self.t>self.c:
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
            V=np.einsum('ij, jk, lk-> il', newX, self.D, newX) + sigma**2 * np.eye(m)
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
    def __init__(self, d, K, alpha, true_beta, c=20):
        
        self.d = d
        self.K = K
        self.true_beta = true_beta      
        self.regrets=[]
        self.beta_errs = []
        
        self.f = np.zeros(d)
        self.B = np.zeros((d,d))
        self.Binv = np.zeros((d,d))
        self.t = 0
        self.c = c
        
        #initival value
        self.beta_hat=np.zeros(d)
        self.D_hat = np.eye(d)
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

        if self.t>self.c:
            self.action=np.random.choice(np.where(est == est.max())[0])
        else:
            self.action=np.random.choice(self.K)
        self.contexts=contexts
        self.regrets.append(np.max(np.sum(contexts @ self.true_beta, axis=1)) - np.sum(contexts @ self.true_beta, axis=1)[self.action])
        return(self.action)

    def update(self,reward):

        newX = self.contexts[self.action]
        self.X.append(self.contexts[self.action])
        self.Y.append(reward)
        
        X = np.array(self.X) 
        Y = np.array(self.Y)
        
        m = len(reward)
                   
        self.flag = (self.c<=self.t<100 and self.t%10==0)+0

        if self.flag==1:           
            _, self.D_hat, self.sigmasq_hat = EM(X,Y,X,beta_init=self.beta_hat, D_init=self.D_hat,sigmasq_init=self.sigmasq_hat,loop=5)

            V=np.einsum('Tij, jk, Tlk-> Til', X, self.D_hat, X) + self.sigmasq_hat * np.eye(m)
            Vinv = np.linalg.inv(V)

            self.f=np.einsum('Tji, Tjk, Tk->i', X, Vinv, Y)
            self.B= np.einsum('Tji, Tjk, Tkl->il', X, Vinv, X)            

        else:           
            V=np.einsum('ij, jk, lk-> il', newX, self.D_hat, newX) + self.sigmasq_hat * np.eye(m)
            Vinv = np.linalg.inv(V)
            
            self.f+=np.einsum('ji, jk, k->i', newX, Vinv, reward)
            self.B+= np.einsum('ji, jk, kl->il', newX, Vinv, newX)
            
        self.Binv = np.linalg.inv(self.B)    
        self.beta_hat=np.dot(self.Binv, self.f)
        self.beta_errs.append(np.sqrt(np.sum((self.beta_hat - self.true_beta)**2)))

def eval_MECUCB1(d, K, m, D=1.0, alpha_set=[0.01, 0.1,1.0], T=1000, M=10, c=10, seed=0):

    results = []
    np.random.seed(seed)
    beta = np.random.uniform(-1,1,d)/np.sqrt(d)
    D=D*np.ones((d,d))

    for alpha in alpha_set:
        
        regrets = np.zeros((M,T))
        errs = np.zeros((M,T))
        
        for repeat in range(M):
            print('MECUCB1 Simulation %d, d=%d, K=%d, m=%d, D=%.1f, alpha=%.2f' % (repeat+1, d, K, m, D[0,0], alpha))

            model = MEUCB(d=d, K=K, D=D, alpha=alpha, true_beta = beta, method='True', c=c)
            
            np.random.seed(seed+repeat)           

            for t in trange(T):
                
                contexts = np.random.randn(K,m,d)
                clip=1./np.maximum(np.ones((K,m)),np.linalg.norm(contexts,axis=2))
                contexts =np.einsum('ijk,ij->ijk',contexts, clip)
                gamma = np.random.multivariate_normal(mean=np.zeros(d), cov=D)
                eps =  sigma * np.random.randn(K,m)

                rewards = contexts @ (beta + gamma) + eps
                ac = model.select_ac(contexts)
                reward = rewards[ac]

                model.update(reward)
                
            regrets[repeat] = np.cumsum(model.regrets)
            errs[repeat] = model.beta_errs

        results.append({'model':'MECUCB1',
                        'alpha':alpha,
                        'regrets':regrets.tolist(),
                        'errs':errs.tolist()})

    with open('./results/d=%d_K=%d_m=%d_D=%.1f_T=%d_c=%d_M=%d/MECUCB1.txt' % (d, K, m, D[0,0], T, c, M), 'w') as outfile:
        json.dump(results, outfile)
        
def eval_C2UCB(d, K, m, D=1.0, alpha_set=[0.01, 0.1,1.0], T=1000, M=10, c=10, seed=0):

    results = []
    np.random.seed(seed)
    beta = np.random.uniform(-1,1,d)/np.sqrt(d)
    D=D*np.ones((d,d))

    for alpha in alpha_set:
        
        regrets = np.zeros((M,T))
        errs = np.zeros((M,T))
        
        for repeat in range(M):
            print('C2UCB Simulation %d, d=%d, K=%d, m=%d, D=%.1f, alpha=%.2f' % (repeat+1, d, K, m, D[0,0], alpha))

            model = MEUCB(d=d, K=K, D=D, alpha=alpha, true_beta = beta, method='indep', c=c)
            
            np.random.seed(seed+repeat)           

            for t in trange(T):
                
                contexts = np.random.randn(K,m,d)
                clip=1./np.maximum(np.ones((K,m)),np.linalg.norm(contexts,axis=2))
                contexts =np.einsum('ijk,ij->ijk',contexts, clip)
                gamma = np.random.multivariate_normal(mean=np.zeros(d), cov=D)
                eps =  sigma * np.random.randn(K,m)
                rewards = contexts @ (beta + gamma) + eps
                ac = model.select_ac(contexts)
                reward = rewards[ac]

                model.update(reward)
                
            regrets[repeat] = np.cumsum(model.regrets)
            errs[repeat] = model.beta_errs

        results.append({'model':'C2UCB',
                        'alpha':alpha,
                        'regrets':regrets.tolist(),
                        'errs':errs.tolist()})

    with open('./results/d=%d_K=%d_m=%d_D=%.1f_T=%d_c=%d_M=%d/C2UCB.txt' % (d, K, m, D[0,0], T, c, M), 'w') as outfile:
        json.dump(results, outfile)
        
def eval_MECUCB2(d, K, m, D=1.0, alpha_set=[0.01, 0.1,1.0], T=1000, M=10, c=10, seed=0):
    #evaluate UCB
    results = []
    np.random.seed(seed)
    beta = np.random.uniform(-1,1,d)/np.sqrt(d)
    D=D*np.eye(d)

    for alpha in alpha_set:
        
        regrets = np.zeros((M,T))
        errs = np.zeros((M,T))
        
        for repeat in range(M):
            print('MECUCB2 Simulation %d, d=%d, K=%d, m=%d, D=%.1f, alpha=%.2f' % (repeat+1, d, K, m, D[0,0], alpha))
            # call model
            model = MEUCB_update(d=d, K=K, alpha=alpha, true_beta = beta, c=c)
            
            np.random.seed(seed+repeat)           

            for t in trange(T):
                
                contexts = np.random.randn(K,m,d)
                clip=1./np.maximum(np.ones((K,m)),np.linalg.norm(contexts,axis=2))
                contexts = np.einsum('ijk,ij->ijk',contexts, clip)
                gamma = np.random.multivariate_normal(mean=np.zeros(d), cov=D)
                eps =  sigma * np.random.randn(K,m)
                rewards = contexts @ (beta + gamma) + eps
                ac = model.select_ac(contexts)
                reward = rewards[ac]

                model.update(reward)
                
            regrets[repeat] = np.cumsum(model.regrets)
            errs[repeat] = model.beta_errs

        results.append({'model':'MECUCB2',
                        'alpha':alpha,
                        'regrets':regrets.tolist(),
                        'errs':errs.tolist()})

    with open('./results/d=%d_K=%d_m=%d_D=%.1f_T=%d_c=%d_M=%d/MECUCB2.txt' % (d, K, m, D[0,0], T, c, M), 'w') as outfile:
        json.dump(results, outfile)