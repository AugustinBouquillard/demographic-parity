import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

# NB: trouver les quantiles, puis pour évaluer dans les autres groupes interpoler avec la fonction np.interp

class OTAwareFairRegressor:
    """
    Optimal Fair Regressor using the Wasserstein 2 barycenter between conditional Y distributions depending on their sensitive attributes.
    We first train a base estimator (unfair) and then correct its outputs to satisfy the Demographic Parity criterion following the method from Chzhen et al. "Fair Regression with Wasserstein Barycenters".
    """
    def __init__(self, base_estimator_model, proxy_estimator=None, sigma=1e-5):
        self.base_estimator = clone(base_estimator_model) #unfitted machine learning model (for instance a RandomForestRegressor()).
        self.sigma = sigma #jitter parameter for uniform noise to break ties, which is required according to Chzhen et al.
        self.p_hat = {}
        self.ar0 = {}
        self.ar1 = {}
        self.groups = []
        if proxy_estimator is None:
            self.proxy_estimator = LogisticRegression()
        else:
            self.proxy_estimator = proxy_estimator #unfitted proxy estimator for estimating group-specific conditional expectations, also called DELTA.

    def fit(self, X_train, y_train, S_train, X_unlabeled=None, S_unlabeled=None):
        """
        Trains the base estimator and calibrates the Wasserstein fair transformation.
        """
        self.proxy_estimator.fit(X_train, S_train)
        X_S_train_combined = np.column_stack((X_train, S_train))
        self.base_estimator.fit(X_S_train_combined, y_train) 
        X_calib = X_train if X_unlabeled is None else X_unlabeled
        S_calib = S_train if S_unlabeled is None else S_unlabeled

        self.groups, counts = np.unique(S_calib, return_counts=True)
        n_total = len(S_calib)
        
        #empirical frequencies 
        self.p_hat = {s: count / n_total for s, count in zip(self.groups, counts)}
        
        for s in self.groups:
            X_s = X_calib[S_calib == s]
            #splitting data into 2 parts
            half = len(X_s) // 2
            X_s_0, X_s_1 = X_s[:half], X_s[half:]
            XS_0 = np.column_stack((X_s_0, np.full(len(X_s_0), s)))
            XS_1 = np.column_stack((X_s_1, np.full(len(X_s_1), s)))
            #predicting and applying uniform jitter to break potential ties 
            pred_0 = self.base_estimator.predict(XS_0)
            pred_1 = self.base_estimator.predict(XS_1)
            
            ar0_s = pred_0 + np.random.uniform(-self.sigma, self.sigma, size=len(pred_0))
            ar1_s = pred_1 + np.random.uniform(-self.sigma, self.sigma, size=len(pred_1))
            self.ar0[s] = np.sort(ar0_s)
            self.ar1[s] = np.sort(ar1_s)
            
        return self

    def predict(self, X, S=None):
        predictions = np.zeros(len(X))
        #if S is missing, we use the proxy estimator to guess the classes
        if S is None: #this is the awareness-derived case with S hat plugged in
            S = self.proxy_estimator.predict(X)

        for s in self.groups:
            mask = (S == s)
            if not np.any(mask): 
                continue
            
            #base values
            XS = np.column_stack((X[mask], S[mask]))
            f_val = self.base_estimator.predict(XS)
            f_val += np.random.uniform(-self.sigma, self.sigma, size=np.sum(mask))
            
            #searchsort to find the rank
            k_s = np.searchsorted(self.ar1[s], f_val)
            
            #converting the rank into a quantile
            q = k_s / len(self.ar1[s])
            
            #computing one-d barycenter 
            g_hat = np.zeros(len(f_val))
            for s_prime in self.groups:
                ar0_sp = self.ar0[s_prime]
                n_sp = len(ar0_sp)
                
                target_q = np.linspace(0, 1, n_sp)
                mapped_values = np.interp(q, target_q, ar0_sp) #evaluating the target values (ar0_sp) at the specific quantiles (q) with np.interp
                
                g_hat += self.p_hat[s_prime] * mapped_values

            predictions[mask] = g_hat
            
        return predictions
    

        #the folloxing was an attempt of extending the plug in method to take into account not only the sign of DELTA but also its magnitude, i.e. using probabilities of belonging to each group.
        #we ended up not using it.
        """
        def predict(self, X, S=None):
        predictions = np.zeros(len(X))

        if S is not None:
            # AWARENESS CONTEXT
            for s in self.groups:
                mask = (S == s)
                if not np.any(mask): 
                    continue
                
                #XS = np.column_stack((X[mask], S[mask]))
                #f_val = self.base_estimator.predict(XS)
                #f_val += np.random.uniform(-self.sigma, self.sigma, size=np.sum(mask))
                
                #k_s = np.searchsorted(self.ar1[s], f_val)
                
                #calculating barycenter
                #g_hat = np.zeros(np.sum(mask))
                #for s_prime in self.groups:
                #    ar0_sp = self.ar0[s_prime]
                #    n_sp = len(ar0_sp)
                #    
                #    # Computing mapped indices
                #    idx = (n_sp * k_s) // len(self.ar1[s])
                #    idx = np.clip(idx, 0, n_sp - 1) #to prevent potential out-of-bound errors
                #    
                #    g_hat += self.p_hat[s_prime] * ar0_sp[idx]
                
                #predictions[mask] = g_hat
                
                
                #searchsorting to find the rank
                k_s = np.searchsorted(self.ar1[s], f_val)
                
                #converting the rank into a quantile (percentage between 0.0 and 1.0)
                q = k_s / len(self.ar1[s])
                
                g_hat = np.zeros(len(f_val))
                for s_prime in self.groups:
                    ar0_sp = self.ar0[s_prime]
                    n_sp = len(ar0_sp)
                    
                    # Create a theoretical grid of quantiles for the target group which maps each sorted value in ar0_sp to a percentile between 0 and 1
                    target_q = np.linspace(0, 1, n_sp)
                    
                    # We evaluate the target values (ar0_sp) at the specific quantiles (q) with np.interp 
                    mapped_values = np.interp(q, target_q, ar0_sp)
                    
                    g_hat += self.p_hat[s_prime] * mapped_values

                predictions[mask] = g_hat
        else:
            # UNAWARENESS CONTEXT: possibility of using probabilities of belonging to each group if given by the proxy estimator (DELTA)
            delta = self.proxy_estimator.predict_proba(X)
            expected_g = np.zeros(len(X))
            
            for j, s in enumerate(self.proxy_estimator.classes_):
                if s not in self.groups:
                    continue
                
                # Assuming all instances belong to group 's' to find what their prediction would be if that were truly the case in the awareness framework.
                XS = np.column_stack((X, np.full(len(X), s)))
                f_val = self.base_estimator.predict(XS)
                f_val += np.random.uniform(-self.sigma, self.sigma, size=len(X))
                
                
                #k_s = np.searchsorted(self.ar1[s], f_val)
                
                #g_hat_s = np.zeros(len(X))
                #for s_prime in self.groups:
                #    ar0_sp = self.ar0[s_prime]
                #    n_sp = len(ar0_sp)
                    
                #    idx = (n_sp * k_s) // len(self.ar1[s])
                #    idx = np.clip(idx, 0, n_sp - 1)
                    
                #    g_hat_s += self.p_hat[s_prime] * ar0_sp[idx]

                
                #finding the rank
                k_s = np.searchsorted(self.ar1[s], f_val)
                
                #making the rank into a quantile (percentage between 0.0 and 1.0)
                q = k_s / len(self.ar1[s])
                
                g_hat_s = np.zeros(len(f_val))
                for s_prime in self.groups:
                    ar0_sp = self.ar0[s_prime]
                    n_sp = len(ar0_sp)
                    
                    target_q = np.linspace(0, 1, n_sp)
                    
                    #We evaluate the target values (ar0_sp) at the specific quantiles (q) with np.interp 
                    mapped_values = np.interp(q, target_q, ar0_sp)
                    
                    g_hat_s += self.p_hat[s_prime] * mapped_values
                
                #multiplying the hypothetical fair prediction by the probability that the point actually belongs to group 's'
                expected_g += delta[:, j] * g_hat_s
            
                
            predictions = expected_g
            
        return predictions
        """