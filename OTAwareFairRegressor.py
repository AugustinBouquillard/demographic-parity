import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

#trouver les quantiles, interpoler cf. fonction interp en python

class OTAwareFairRegressor:
    """
    Optimal Fair Regressor using Wasserstein Barycenters.
    Trains a base estimator and transforms its outputs to satisfy Demographic Parity following the method from Chzhen et al. "Fair Regression with Wasserstein Barycenters".
    """
    def __init__(self, base_estimator_model, proxy_estimator=None, sigma=1e-5):
        """
        base_estimator_model: an unfitted machine learning model (e.g., RandomForestRegressor()).
        proxy_estimator: also known as DELTA. An unfitted proxy estimator for estimating group-specific conditional expectations.
        sigma: jitter parameter for uniform noise to break ties.
        """
        # Clone the model to ensure we are starting with a fresh, unfitted estimator
        self.base_estimator = clone(base_estimator_model)
        self.sigma = sigma
        self.p_hat = {}
        self.ar0 = {}
        self.ar1 = {}
        self.groups = []
        if proxy_estimator is None:
            self.proxy_estimator = LogisticRegression()
        else:
            self.proxy_estimator = proxy_estimator

    def fit(self, X_train, y_train, S_train, X_unlabeled=None, S_unlabeled=None):
        """
        Trains the base estimator and calibrates the Wasserstein fair transformation.
        """
        self.proxy_estimator.fit(X_train, S_train)

        # 1. Train the base estimator \hat{f} on labeled data 
        X_S_train_combined = np.column_stack((X_train, S_train))
        self.base_estimator.fit(X_S_train_combined, y_train)

        # 2. Setup unlabeled data \mathcal{U} for calibration 
        # Fallback to training data if no distinct unlabeled pool is provided
        X_calib = X_train if X_unlabeled is None else X_unlabeled
        S_calib = S_train if S_unlabeled is None else S_unlabeled

        self.groups, counts = np.unique(S_calib, return_counts=True)
        n_total = len(S_calib)
        
        # Estimate empirical frequencies \hat{p}_s 
        self.p_hat = {s: count / n_total for s, count in zip(self.groups, counts)}
        
        # 3. Performing the group-wise calibration 
        for s in self.groups:
            # Isolating unlabeled data for group s 
            X_s = X_calib[S_calib == s]
            
            # Splitting data into two equal parts
            half = len(X_s) // 2
            X_s_0, X_s_1 = X_s[:half], X_s[half:]
            
            # Re-attaching the sensitive attribute 's' so the base model can predict
            XS_0 = np.column_stack((X_s_0, np.full(len(X_s_0), s)))
            XS_1 = np.column_stack((X_s_1, np.full(len(X_s_1), s)))
            
            # Predicting and apply uniform jitter
            pred_0 = self.base_estimator.predict(XS_0)
            pred_1 = self.base_estimator.predict(XS_1)
            
            ar0_s = pred_0 + np.random.uniform(-self.sigma, self.sigma, size=len(pred_0))
            ar1_s = pred_1 + np.random.uniform(-self.sigma, self.sigma, size=len(pred_1))
            self.ar0[s] = np.sort(ar0_s)
            self.ar1[s] = np.sort(ar1_s)
            
        return self

    def predict(self, X, S=None):
        predictions = np.zeros(len(X))
        # If S is missing, use the proxy estimator to guess the classes
        if S is None:
            S = self.proxy_estimator.predict(X)

        # AWARENESS CONTEXT (Also handles Hard-Prediction Unawareness)
        for s in self.groups:
            mask = (S == s)
            if not np.any(mask): 
                continue
            
            # Predicting base values for all items in this group at once
            XS = np.column_stack((X[mask], S[mask]))
            f_val = self.base_estimator.predict(XS)
            f_val += np.random.uniform(-self.sigma, self.sigma, size=np.sum(mask))
            
            # Vectorized searchsorted to find the rank
            k_s = np.searchsorted(self.ar1[s], f_val)
            
            # Convert the rank into a quantile (percentage between 0.0 and 1.0)
            q = k_s / len(self.ar1[s])
            
            # Calculating Barycenter mapping
            g_hat = np.zeros(len(f_val))
            for s_prime in self.groups:
                ar0_sp = self.ar0[s_prime]
                n_sp = len(ar0_sp)
                
                # Create a theoretical grid of quantiles for the target group
                target_q = np.linspace(0, 1, n_sp)
                
                # We evaluate the target values (ar0_sp) at the specific quantiles (q) with np.interp 
                mapped_values = np.interp(q, target_q, ar0_sp)
                
                g_hat += self.p_hat[s_prime] * mapped_values

            predictions[mask] = g_hat
            
        return predictions
    

        """
        def predict(self, X, S=None):

        #Generates fair predictions using vectorized optimal transport mapping.
        
        #X: Input features.
        #S: Exact sensitive attributes (for the Awareness context). 
        #   If None, uses the proxy estimator (Unawareness context).

        predictions = np.zeros(len(X))

        if S is not None:
            # AWARENESS CONTEXT
            for s in self.groups:
                mask = (S == s)
                if not np.any(mask): 
                    continue
                
                # Predicting base values for all items in this group at once
                #XS = np.column_stack((X[mask], S[mask]))
                #f_val = self.base_estimator.predict(XS)
                #f_val += np.random.uniform(-self.sigma, self.sigma, size=np.sum(mask))
                
                # Vectorized searchsorted
                #k_s = np.searchsorted(self.ar1[s], f_val)
                
                # Calculating Barycenter mapping
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
                
                
                # Vectorized searchsorted to find the rank
                k_s = np.searchsorted(self.ar1[s], f_val)
                
                # Convert the rank into a quantile (percentage between 0.0 and 1.0)
                q = k_s / len(self.ar1[s])
                
                # Calculating Barycenter mapping
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
            
            # Looping through the classes that the proxy estimator learned
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

                
                # Vectorized searchsorted to find the rank
                k_s = np.searchsorted(self.ar1[s], f_val)
                
                # Convert the rank into a quantile (percentage between 0.0 and 1.0)
                q = k_s / len(self.ar1[s])
                
                # Calculating Barycenter mapping
                g_hat_s = np.zeros(len(f_val))
                for s_prime in self.groups:
                    ar0_sp = self.ar0[s_prime]
                    n_sp = len(ar0_sp)
                    
                    target_q = np.linspace(0, 1, n_sp)
                    
                    # We evaluate the target values (ar0_sp) at the specific quantiles (q) with np.interp 
                    mapped_values = np.interp(q, target_q, ar0_sp)
                    
                    g_hat_s += self.p_hat[s_prime] * mapped_values
                
                # Multiplying the hypothetical fair prediction by the probability that the point actually belongs to group 's'
                expected_g += delta[:, j] * g_hat_s
            
                
            predictions = expected_g
            
        return predictions
        """