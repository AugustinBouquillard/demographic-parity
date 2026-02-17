import numpy as np

class WassersteinFairRegressor:
    """
    Optimal Fair Regressor using Wasserstein Barycenters.
    Transforms an unfair base estimator into a fair one satisfying Demographic Parity.
    """
    def __init__(self, base_estimator, sigma=1e-5):
        """
        base_estimator: A trained model with a `.predict(X, S)` method.
        sigma: Jitter parameter for uniform noise to break ties.
        """
        self.base_estimator = base_estimator
        self.sigma = sigma
        self.p_hat = {}
        self.ar0 = {}
        self.ar1 = {}
        self.groups = []

    def fit_transform_unlabeled(self, X_unlabeled, S_unlabeled):
        """
        Precomputes the empirical CDFs and quantile functions using unlabeled data.
        """
        self.groups = np.unique(S_unlabeled)
        n_total = len(S_unlabeled)
        
        for s in self.groups:
            # 1. Get unlabeled data for group s
            mask = (S_unlabeled == s)
            X_s = X_unlabeled[mask]
            
            # 2. Empirical frequencies
            self.p_hat[s] = len(X_s) / n_total
            
            # 3. Split data in two equal parts as dictated by the algorithm
            n_s = len(X_s)
            half = n_s // 2
            X_s_0 = X_s[:half]
            X_s_1 = X_s[half:]
            
            S_s_0 = np.full(len(X_s_0), s)
            S_s_1 = np.full(len(X_s_1), s)
            
            # 4. Generate base predictions and apply uniform jitter
            pred_0 = self.base_estimator.predict(X_s_0, S_s_0)
            pred_1 = self.base_estimator.predict(X_s_1, S_s_1)
            
            ar0_s = pred_0 + np.random.uniform(-self.sigma, self.sigma, size=len(pred_0))
            ar1_s = pred_1 + np.random.uniform(-self.sigma, self.sigma, size=len(pred_1))
            
            # 5. Sort arrays for fast evaluation of empirical CDF and Quantiles
            self.ar0[s] = np.sort(ar0_s)
            self.ar1[s] = np.sort(ar1_s)

    def _predict_single(self, x, s):
        """
        Computes the fair prediction g_hat(x, s) for a single instance.
        """
        # Get base prediction and add jitter
        x_array = np.array([x])
        s_array = np.array([s])
        f_val = self.base_estimator.predict(x_array, s_array)[0]
        f_val += np.random.uniform(-self.sigma, self.sigma)
        
        # Evaluate empirical CDF: position of f_val in ar1_s
        ar1_s = self.ar1[s]
        k_s = np.searchsorted(ar1_s, f_val)
        
        g_hat = 0.0
        # Calculate the barycenter mapping 
        for s_prime in self.groups:
            ar0_sp = self.ar0[s_prime]
            n_sp = len(ar0_sp)
            
            # Find matching quantile index in ar0_s'
            idx = int((n_sp * k_s) / len(ar1_s))
            idx = min(idx, n_sp - 1) # Handle edge case
            
            g_hat += self.p_hat[s_prime] * ar0_sp[idx]
            
        return g_hat
        
    def predict(self, X, S=None, delta=None):
        """
        Generates fair predictions.
        
        X: Input features.
        S: Exact sensitive attributes (for the Awareness context).
        delta: Matrix of shape (n_samples, n_groups) with estimated probabilities 
               of S (for the Unawareness context).
        """
        if S is None and delta is None:
            raise ValueError("Must provide either true 'S' or estimated 'delta'.")
            
        predictions = []
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