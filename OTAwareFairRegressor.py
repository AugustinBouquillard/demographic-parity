import numpy as np

import numpy as np
from sklearn.base import clone
from sklearn.linear_model import LogisticRegression

#trouver les quantiles interpoler cf. fonction interp en python

class OTAwareFairRegressor:
    """
    Optimal Fair Regressor using Wasserstein Barycenters.
    Trains a base estimator and transforms its outputs to satisfy Demographic Parity.
    """
    def __init__(self, base_estimator_model, proxy_estimator=None, sigma=1e-5):
        """
        base_estimator_model: An unfitted machine learning model (e.g., RandomForestRegressor()).
        proxy_estimator: also known as DELTA. An unfitted proxy estimator for estimating group-specific conditional expectations.
        sigma: Jitter parameter for uniform noise to break ties.
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

    import numpy as np

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
        
        # 3. Perform the group-wise calibration (Algorithm 1) 
        for s in self.groups:
            # Isolate unlabeled data \mathcal{U}^s for group s 
            X_s = X_calib[S_calib == s]
            
            # Split data into two equal parts: \mathcal{U}_0^s and \mathcal{U}_1^s 
            half = len(X_s) // 2
            X_s_0, X_s_1 = X_s[:half], X_s[half:]
            
            # Re-attach the sensitive attribute 's' so the base model can predict
            XS_0 = np.column_stack((X_s_0, np.full(len(X_s_0), s)))
            XS_1 = np.column_stack((X_s_1, np.full(len(X_s_1), s)))
            
            # Predict and apply uniform jitter U([-\sigma, \sigma]) 
            pred_0 = self.base_estimator.predict(XS_0)
            pred_1 = self.base_estimator.predict(XS_1)
            
            ar0_s = pred_0 + np.random.uniform(-self.sigma, self.sigma, size=len(pred_0))
            ar1_s = pred_1 + np.random.uniform(-self.sigma, self.sigma, size=len(pred_1))
            
            # Sort arrays (ar_0^{s'} and ar_1^{s'}) for fast evaluation 
            self.ar0[s] = np.sort(ar0_s)
            self.ar1[s] = np.sort(ar1_s)
            
        return self

    def _predict_single(self, x, s):
        """
        Computes the fair prediction g_hat(x, s) for a single instance[cite: 131].
        """
        # Combine x and s for the base estimator
        x_s_combined = np.concatenate((x, [s])).reshape(1, -1)
        f_val = self.base_estimator.predict(x_s_combined)[0]
        
        # Add jitter [cite: 131]
        f_val += np.random.uniform(-self.sigma, self.sigma)
        
        # Evaluate empirical CDF
        ar1_s = self.ar1[s]
        k_s = np.searchsorted(ar1_s, f_val)
        
        g_hat = 0.0
        # Calculate the barycenter mapping [cite: 131]
        for s_prime in self.groups:
            ar0_sp = self.ar0[s_prime]
            n_sp = len(ar0_sp)
            
            idx = int((n_sp * k_s) / len(ar1_s))
            idx = min(idx, n_sp - 1)
            
            g_hat += self.p_hat[s_prime] * ar0_sp[idx]
            
        return g_hat
        
    def predict(self, X, S=None):
        """
        Generates fair predictions.
        
        X: Input features.
        S: Exact sensitive attributes (for the Awareness context).
        delta: Matrix of shape (n_samples, n_groups) with estimated probabilities 
               of S (for the Unawareness context).
        """
        predictions = []

        if S is None :
            delta=self.proxy_estimator.predict_proba(X)
         
        for i in range(len(X)):
            x = X[i]
            
            if S is not None:
                # Awareness Context: Use the known sensitive attribute
                predictions.append(self._predict_single(x, S[i]))
                
            elif delta is not None:
                # Unawareness Context: Expected prediction using probability estimates
                expected_g = 0.0
                for j, s_prime in enumerate(self.groups):
                    prob = delta[i][j]
                    if prob > 0:
                        expected_g += prob * self._predict_single(x, s_prime)
                predictions.append(expected_g)
                
        return np.array(predictions)