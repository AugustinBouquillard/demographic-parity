import numpy as np
from scipy.special import softmax
import collections
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error as mse

from evaluation_measures import DP_unfairness, prob_unfairness, unfairness

class FairReg:
    def __init__(self, base_method, classifier, B, K, p, eps, T, keep_history=False):
        self.base_method = base_method
        self.classifier = classifier
        self.B = B #bound on target
        self.K = K #number of sensitive attributes
        self.p = p #frequencies
        self.eps = eps #epsilon threshold
        self.T = T #number of iterations
        self.keep_history = keep_history #keeping history of estimators

    def stoch_grad(self, w, x):
        self.stoch_grad_counter += 1

        reg_pred = self.base_method.predict(x.reshape(1, -1))
        clf_prob = self.classifier.predict_proba(x.reshape(1, -1))
        tau_x_coef = np.zeros(clf_prob.shape)
        for i, p_i in enumerate(self.p):
            tau_x_coef[:,i] = 1 - clf_prob[:,i]/p_i 

        r_x = np.square(reg_pred[:, np.newaxis] - self.Q_L)
        
        #computing the shtochastic gradient
        grad = np.zeros(2*self.K*(2*self.L+1))
        diff = (w[:self.K*(2*self.L+1)].copy() - w[self.K*(2*self.L+1):].copy()).reshape((self.K,2*self.L+1))
        softmax_x = softmax(self.beta*(np.matmul(tau_x_coef,diff) - r_x), axis = 1)
        for i in range(self.K):
            grad[i*(2*self.L+1):(i+1)*(2*self.L+1)] = np.mean(tau_x_coef[:,i][:, None]*softmax_x,axis=0) + self.eps[i]
            grad[(i+self.K)*(2*self.L+1):(i+1+self.K)*(2*self.L+1)] = -np.mean(tau_x_coef[:,i][:, None]*softmax_x,axis=0) + self.eps[i]
        return grad
    
    def stoch_grad_reg(self, w, x, w_reg=[], mu_reg=[]):
        reg = 0
        for i in range(len(w_reg)):
            reg += mu_reg[i]*(w-w_reg[i])
        return self.stoch_grad(w, x) + reg

    def SGD(self, X, w_init, alpha, T, w_reg=[], mu_reg=[]):
        w = w_init
        w_all = []
        w_hist = []
        for t in range(T):
            x = X[np.random.randint(len(X))]  
            w -= alpha * self.stoch_grad_reg(w, x, w_reg, mu_reg)
            w[w<0] = 0
            w_all.append(w)
            if self.keep_history:
                w_hist.append(np.mean(w_all, axis=0))

        return np.mean(w_all, axis=0), w_hist  
    
    def SGD_sc(self, X, w_init, mu, M, T, w_reg=[], mu_reg=[]):
        N1 = int(np.floor(T / (M / mu)))
        K1 = int(np.floor(np.log2(mu * T / M)))
        w = w_init
        w_hist = []
        for t in range(0, N1):
            w, w_hist_ = self.SGD(X, w, 1 / (2 * M), int(np.floor(M / mu)), w_reg, mu_reg)
            if self.keep_history:
                w_hist+=w_hist_
        for k in range(0, K1):
            w, w_hist_ = self.SGD(X, w, 1 / (2**k * M), int(np.floor((2**(k + 2)) * M / mu)), w_reg, mu_reg)
            if self.keep_history:
                w_hist+=w_hist_
        return w, w_hist
    
    def AC_SA(self, X, w_init, mu, M, T, w_reg=[], mu_reg=[]):
        w = w_init
        w_ag = w_init
        w_hist = []
        
        for i in range(T):
            alpha = 2 / (i+1)
            gamma = 4 * M / ((i+1) * (i+2))
            alpha1 = 1 - alpha

            coef1 = (alpha1 * (mu + gamma)) / (gamma + (1 - alpha ** 2) * mu)
            coef2 = alpha * (alpha1 * mu + gamma) / (gamma + (1 - alpha ** 2) * mu)
            coef3 = alpha * mu / (mu + gamma)
            coef4 = (alpha1 * mu + gamma) / (mu + gamma)
            coef5 = alpha / (mu + gamma)

            w_md = coef1 * w_ag + coef2 * w
            x = X[np.random.randint(len(X))]
            w = coef3 * w_md + coef4 * w - coef5 * self.stoch_grad_reg(w_md, x, w_reg, mu_reg)
            w[w<0] = 0
            w_ag = alpha * w + alpha1 * w_ag
            
            if self.keep_history:
                w_hist.append(w_ag)
                
        return w_ag, w_hist
    
    
    def AC_SA_2(self, X, w_init, mu, M, T, w_reg=[], mu_reg=[]):
        
        w_hist = []

        w_ag_1, w_hist_ = self.AC_SA(X, w_init, mu, M, T, w_reg, mu_reg)
        if self.keep_history:
            w_hist+=w_hist_
        w_ag_2, w_hist_ = self.AC_SA(X, w_ag_1, mu, M, T, w_reg, mu_reg)
        if self.keep_history:
            w_hist+=w_hist_
            
        return w_ag_2, w_hist
        
        
    def accelerated_grad(self, X, w_init, mu, M, T, w_reg=[], mu_reg=[]):
        w = w_init
        w_ag = w_init
        w_hist = []
        half_time = int(T / 2)
        for i in range(half_time):
            alpha = 2 / (i+1)
            gamma = 4 * M / ((i+1) * (i+2))
            alpha1 = 1 - alpha

            coef1 = (alpha1 * (mu + gamma)) / (gamma + (1 - alpha ** 2) * mu)
            coef2 = alpha * (alpha1 * mu + gamma) / (gamma + (1 - alpha ** 2) * mu)
            coef3 = alpha * mu / (mu + gamma)
            coef4 = (alpha1 * mu + gamma) / (mu + gamma)
            coef5 = alpha / (mu + gamma)

            w_md = coef1 * w_ag + coef2 * w
            x = X[np.random.randint(len(X))]
            w = coef3 * w_md + coef4 * w - coef5 * self.stoch_grad_reg(w_md, x, w_reg, mu_reg)
            w[w<0] = 0
            w_ag = alpha * w + alpha1 * w_ag
            
            if self.keep_history:
                w_hist.append(w_ag)

        for i in range(half_time):
            alpha = 2 / (i+1)
            gamma = 4 * M / ((i+1) * (i+2))
            alpha1 = 1 - alpha

            coef1 = (alpha1 * (mu + gamma)) / (gamma + (1 - alpha ** 2) * mu)
            coef2 = alpha * (alpha1 * mu + gamma) / (gamma + (1 - alpha ** 2) * mu)
            coef3 = alpha * mu / (mu + gamma)
            coef4 = (alpha1 * mu + gamma) / (mu + gamma)
            coef5 = alpha / (mu + gamma)

            w_md = coef1 * w_ag + coef2 * w
            x = X[np.random.randint(len(X))]
            w = coef3 * w_md + coef4 * w - coef5 * self.stoch_grad_reg(w_md, x, w_reg, mu_reg)
            w[w<0] = 0
            w_ag = alpha * w + alpha1 * w_ag
            
            if self.keep_history:
                w_hist.append(w_ag)

        return w_ag, w_hist


    def SGD3_sc(self, X, w_init, mu_init, M, T, method, sc = True):
        w = w_init
        new_init = w_init
        mu = mu_init
        w_reg = []
        mu_reg = []
        w_history = []

        if sc == False:
            w_reg.append(w)
            mu_reg.append(w)
        S1 = int(np.floor(np.log2(M/mu)))
        for s in range(1, S1+1):
            if method=='ACSA':
                w, w_hist = self.AC_SA(X, new_init, mu, M,
                                                  T=int(np.floor(T/S1)), w_reg=w_reg, mu_reg=mu_reg)
            elif method=='ACSA2':
                w, w_hist = self.AC_SA_2(X, new_init, mu, M,
                                                  T=int(np.floor(T/S1)), w_reg=w_reg, mu_reg=mu_reg)
            elif method=='accelerated':
                w, w_hist = self.accelerated_grad(X, new_init, mu, M,
                                                  T=int(np.floor(T/S1)), w_reg=w_reg, mu_reg=mu_reg)
            elif method=='sgd':
                w, w_hist = self.SGD_sc(X, w, mu, 3*M, int(np.floor(T/S1)), w_reg, mu_reg)
            else:
                raise Exception('Method not found.')
            new_init = w_init
            mu = 2*mu
            w_reg.append(w)
            mu_reg.append(mu)
            if self.keep_history:
                w_history+=w_hist
        return w, w_history


    def SGD3(self, X, w_init, mu, M, T, method='accelerated'):
        return self.SGD3_sc(X, w_init, mu, M + mu, T, method, sc = False)

    def fit(self, X, beta = 'auto', L = 'auto', alg={'base':'SGD3', 'method':'ACSA2'}, history = False):
        
        self.N = len(X)
    
        sum_ps=0
        for p_s in self.p:
            sum_ps += (1-p_s)/p_s
        
        if L == 'auto':
            self.L = int(np.sqrt(self.N)) #discretization param
        else:
            self.L = L
        
        if beta == 'auto':
            self.beta = 0.5 * np.sqrt(self.N) * np.log2(self.N) #temperature param in softmax
        else:
            self.beta = beta
            
        self.Q_L = np.arange(-self.L, self.L + 1) * self.B / self.L #the grid
        self.M = 2*self.beta*sum_ps #Lipschits constant
        #self.mu = 2*sum_ps/self.beta #strong-convexity param
        self.mu =  self.M / self.T
        self.w_0 = np.zeros(2*self.K*(2*self.L+1)) #initial point
        self.stoch_grad_counter = 0
        
        if alg['base']=='SGD3':
            if len(alg)>0:
                self.w_est, self.w_est_hist = self.SGD3(X, self.w_0, self.mu, self.M, self.T, method=alg['method'])
            else:
                self.w_est, self.w_est_hist = self.SGD3(X, self.w_0, self.mu, self.M, self.T, method='ACSA2')
        elif alg['base']=='SGD':
            self.w_est, self.w_est_hist = self.SGD(X, self.w_0, 1/self.M, self.T)
        elif alg['base']=='ACSA':
            self.w_est, self.w_est_hist = self.AC_SA(X, self.w_0, self.mu, self.M, self.T)
        elif alg['base']=='ACSA2':
            self.w_est, self.w_est_hist = self.AC_SA_2(X, self.w_0, self.mu, self.M, self.T)
        else:
            raise Exception('Method not found.')
                                                 
    def predict(self, X):
        
        reg_pred = self.base_method.predict(X)
        clf_prob = self.classifier.predict_proba(X)
        
        tau_X_coef = np.zeros(clf_prob.shape)
        for i, p_i in enumerate(self.p):
            tau_X_coef[:,i] = 1 - clf_prob[:,i]/p_i 
        
        r_X = np.square(reg_pred[:, np.newaxis] - self.Q_L)
        
        diff = (self.w_est[:self.K*(2*self.L+1)].copy() - self.w_est[self.K*(2*self.L+1):].copy()).reshape((self.K,2*self.L+1))
        
        pred_prob = softmax(self.beta*(np.matmul(tau_X_coef,diff) - r_X), axis = 1)
        self.pred_prob = pred_prob
        
        return (np.argmax(pred_prob, axis=1)-self.L)*self.B/self.L
    
    
    def history(self, X, S, y, data_scaling=False, scaler=None):
        
        reg_pred = self.base_method.predict(X)
        clf_prob = self.classifier.predict_proba(X)
        
        tau_X_coef = np.zeros(clf_prob.shape)
        for i, p_i in enumerate(self.p):
            tau_X_coef[:,i] = 1 - clf_prob[:,i]/p_i 
            
        r_X = np.square(reg_pred[:, np.newaxis] - self.Q_L)
        #inverse scaling for evaluation
        if data_scaling:
            y = scaler.inverse_transform(y.values.reshape(-1, 1))[:,0]
            grid = scaler.inverse_transform(self.Q_L.reshape(-1, 1))[:,0] 
        else:
            grid = self.Q_L

        r_X_y = np.square(y[:, np.newaxis] - grid)
        
        risk_history = []
        unfairness_history = []
        
        for w_est in self.w_est_hist:
            diff = (w_est[:self.K*(2*self.L+1)].copy() - w_est[self.K*(2*self.L+1):].copy()).reshape((self.K,2*self.L+1))
            pred_prob = softmax(self.beta*(np.matmul(tau_X_coef,diff) - r_X), axis = 1)
            
            risk_history.append(np.mean(np.sum(r_X_y*pred_prob, axis=1))) #probabilistic risk history
            unfairness_history.append(unfairness(pred_prob, S)) #unfairness history
            
        return risk_history, unfairness_history
        
        

