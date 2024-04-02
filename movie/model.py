import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
import json

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
    def __init__(self, K, d, c, m, alpha):
        ## Initialization

        self.K = K
        self.c = c
        self.m = m
        self.outputs = []
        self.beta_hat=np.zeros(d)
        self.f = np.zeros(d)
        self.B = m*np.eye(d)
        self.Binv = (1./m) * np.eye(d)
        self.t = 0
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
        return(self.action)

    def update(self,reward):
        self.outputs.append(reward)
        newX=self.contexts[self.action]
        
        V=np.eye(self.m)
        Vinv=np.eye(self.m)

        self.f+=np.einsum('ji, jk, k->i', newX, Vinv, reward)

        self.B+= np.einsum('ji, jk, kl->il', newX, Vinv, newX)
        self.Binv = np.linalg.inv(self.B)
        self.beta_hat=np.dot(self.Binv, self.f)
        
        
class MEUCB_update:
    def __init__(self, K, d, c, alpha):

        self.K = K
        self.c = c
        
        self.outputs = []
        
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

        if self.t>self.c:
            self.action=np.random.choice(np.where(est == est.max())[0])
        else:
            self.action=np.random.choice(self.K)
        self.contexts=contexts
        return(self.action)

    def update(self,reward):

        self.outputs.append(reward)
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