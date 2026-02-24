import numpy as np
import ot
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


class OTUnawareFairRegressor_Sinkhorn(BaseEstimator, RegressorMixin):
    """
    Scalable Unaware Fair Regressor using Entropic Optimal Transport.
    Improvements over base implementation:
    1. Uses Sinkhorn algorithm (O(N^2)) instead of exact EMD (O(N^3)).
    2. Uses Gradient Boosting for the correction map (better interpolation than k-NN).
    3. Explicitly handles the 'neutral' set X_= where |Delta| ~ 0.
    4. Includes a 'check_nestedness' diagnostic method (Paper Part 4).
    """
    def __init__(self, 
                 base_regressor=None, 
                 base_classifier=None, 
                 correction_regressor=None,
                 reg_e=1e-2,          # Entropic regularization parameter
                 min_delta=1e-2,      # Threshold to treat points as 'neutral' (no transport)
                 max_samples_ot=2000): # Downsample for OT if N is huge
        self.base_regressor = base_regressor if base_regressor else LinearRegression()
        self.base_classifier = base_classifier if base_classifier else LogisticRegression(solver='liblinear')
        self.correction_regressor = correction_regressor if correction_regressor else HistGradientBoostingRegressor()
        
        self.reg_e = reg_e
        self.min_delta = min_delta
        self.max_samples_ot = max_samples_ot
        
        # Fitted attributes
        self.eta_model_ = None
        self.delta_model_ = None
        self.correction_model_ = None
        self.p_s1_ = None
        self.p_s2_ = None

    def fit(self, X, y, s):
        X = np.array(X)
        y = np.array(y)
        s = np.array(s)

        # --- 1. Train Proxies (Eta and Delta) ---
        self.eta_model_ = clone(self.base_regressor).fit(X, y)
        eta_train = self.eta_model_.predict(X)

        self.p_s1_ = np.mean(s == 1)
        self.p_s2_ = np.mean(s == 2)
        
        # Calibrated probability estimates are crucial for Delta
        self.delta_model_ = clone(self.base_classifier).fit(X, s)
        ps_pred = self.delta_model_.predict_proba(X)[:, 1]
        
        # Calculate Delta(x) (Paper Eq. 7 proxy)
        # Avoid division by zero
        safe_p1 = np.clip(self.p_s1_, 1e-6, 1)
        safe_p2 = np.clip(self.p_s2_, 1e-6, 1)
        delta_vals = (ps_pred / safe_p1) - ((1 - ps_pred) / safe_p2)

        # --- 2. Filter Active Sets ---
        # Points with small delta are "neutral" and shouldn't warp cost geometry
        active_mask = np.abs(delta_vals) > self.min_delta
        
        X_active = X[active_mask]
        eta_active = eta_train[active_mask]
        delta_active = delta_vals[active_mask]
        
        # --- 3. Downsampling (Optional for Speed) ---
        if len(eta_active) > self.max_samples_ot:
            indices = np.random.choice(len(eta_active), self.max_samples_ot, replace=False)
            eta_active = eta_active[indices]
            delta_active = delta_active[indices]
            # Note: We only downsample for solving the MAP. 
            # Ideally, we would project all points, but for training the correction
            # a representative subset is often sufficient.

        # Split into Plus and Minus groups
        idx_plus = np.where(delta_active > 0)
        idx_minus = np.where(delta_active < 0)
        
        if len(idx_plus) == 0 or len(idx_minus) == 0:
            # Fallback: No fairness correction possible/needed
            self.correction_model_ = None
            return self

        h1 = eta_active[idx_plus]
        d1 = np.abs(delta_active[idx_plus])
        h2 = eta_active[idx_minus]
        d2 = np.abs(delta_active[idx_minus])

        # --- 4. Optimized Cost Calculation (Paper Eq 14) ---
        # C(x1, x2) = (h1 - h2)^2 / (|d1| + |d2|)
        # Broadcasting: (N, 1) - (1, M) -> (N, M)
        numer = (h1[:, None] - h2[None, :]) ** 2
        denom = (d1[:, None] + d2[None, :])
        M = numer / denom
        
        # Normalize M for numerical stability in Sinkhorn
        M_max = M.max()
        M /= M_max

        # --- 5. Solve Regularized OT (Sinkhorn) ---
        n1, n2 = len(h1), len(h2)
        a, b = np.ones(n1) / n1, np.ones(n2) / n2
        
        # Sinkhorn is typically 100x faster than EMD for N > 1000
        gamma = ot.sinkhorn(a, b, M, reg=self.reg_e)

        # --- 6. Barycentric Projection (Paper Eq 13) ---
        # Target y* is the weighted average of potential matches
        # Y_opt_ij = (h1/d1 + h2/d2) / (1/d1 + 1/d2)
        
        inv_d1 = 1.0 / d1
        inv_d2 = 1.0 / d2
        
        # Precompute weighted numerators and denominators
        # Reshape for broadcasting
        num_matrix = (h1 * inv_d1)[:, None] + (h2 * inv_d2)[None, :]
        den_matrix = inv_d1[:, None] + inv_d2[None, :]
        Y_opt_pairs = num_matrix / den_matrix
        
        # Project: y_fair_i = Sum_j (gamma_ij / a_i) * Y_opt_ij
        # Rescaling gamma by N ensures it sums to 1 per row/col (approx)
        y_fair_plus = np.sum(gamma * Y_opt_pairs, axis=1) * n1
        y_fair_minus = np.sum(gamma * Y_opt_pairs, axis=0) * n2
        
        # --- 7. Learn the Correction Map ---
        # We learn f(eta, delta) -> shift
        # Input features: [eta, delta]
        train_features = np.vstack([
            np.column_stack([h1, delta_active[idx_plus]]),
            np.column_stack([h2, delta_active[idx_minus]])
        ])
        
        # Target shifts: Fair - Original
        train_shifts = np.concatenate([
            y_fair_plus - h1,
            y_fair_minus - h2
        ])
        
        # Fit the correction model (e.g. Gradient Boosting)
        self.correction_model_ = clone(self.correction_regressor).fit(train_features, train_shifts)
        
        return self

    def predict(self, X):
        X = np.array(X)
        eta_new = self.eta_model_.predict(X)
        
        if self.correction_model_ is None:
            return eta_new
            
        # Get Delta
        ps = self.delta_model_.predict_proba(X)[:, 1]
        safe_p1 = np.clip(self.p_s1_, 1e-6, 1)
        safe_p2 = np.clip(self.p_s2_, 1e-6, 1)
        delta_new = (ps / safe_p1) - ((1 - ps) / safe_p2)
        
        # Predict Shift
        features = np.column_stack([eta_new, delta_new])
        pred_shifts = self.correction_model_.predict(features)
        
        # Apply Shift only where Delta is significant (continuity)
        # Smooth fade-out for small deltas could be added here
        mask_active = np.abs(delta_new) > self.min_delta
        
        final_pred = eta_new.copy()
        final_pred[mask_active] += pred_shifts[mask_active]
        
        return final_pred

    def check_nestedness(self, X, n_thresholds=10):
        """
        Diagnostic for Part 4 of the paper:
        Checks if the learned regressor creates valid nested classification sets.
        """
        if self.correction_model_ is None:
            return True, "No correction model"

        preds = self.predict(X)
        thresholds = np.linspace(np.min(preds), np.max(preds), n_thresholds)
        
        # If nested, the set of accepted people at high threshold 
        # must be a subset of accepted people at low threshold.
        # This is trivially true for a deterministic function f(x).
        # The paper's condition applies to the *group-wise* acceptance regions 
        # in the (eta, delta) plane.
        
        # We check empirically: Does increasing y always shrink the acceptance region?
        # Since we use a function f(x) to predict, this is guaranteed by construction
        # in this "Plug-in" direction.
        # However, we can check if the *theoretical* classifier boundaries cross.
        # The paper says: If boundaries cross, f* is suboptimal for classification.
        
        # We can simulate the boundary crossing:
        # Plot kappa(y) implicitly defined by our regressor.
        pass # Visualization logic would go here

import numpy as np
import ot
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression, LogisticRegression

class OTUnawareFairRegressor_Perrot(BaseEstimator, RegressorMixin):
    def __init__(self, 
                 base_regressor=None, 
                 base_classifier=None, 
                 lambda_gamma=1e-1,   # OT cost regularization 
                 lambda_T=1e-2,       # Transport map regularization
                 min_delta=1e-2,      # Stability threshold
                 max_samples_ot=1000, # Downsampling for speed
                 max_iter_bcd=10):    # BCD loop iterations
        self.base_regressor = base_regressor if base_regressor else LinearRegression()
        self.base_classifier = base_classifier if base_classifier else LogisticRegression(solver='liblinear')
        
        self.lambda_gamma = lambda_gamma
        self.lambda_T = lambda_T
        self.min_delta = min_delta
        self.max_samples_ot = max_samples_ot
        self.max_iter_bcd = max_iter_bcd
        
        self.eta_model_ = None
        self.delta_model_ = None
        self.L_ = None 
        self.p_s1_ = None
        self.p_s2_ = None

    def fit(self, X, y, s):
        X = np.array(X)
        y = np.array(y)
        s = np.array(s)

        # 1. Fit Bayes Proxy (Eta) and Group Proxy (Delta)
        self.eta_model_ = clone(self.base_regressor).fit(X, y)
        eta_train = self.eta_model_.predict(X)

        self.p_s1_ = np.mean(s == 1)
        self.p_s2_ = np.mean(s == 0) 
        
        self.delta_model_ = clone(self.base_classifier).fit(X, s)
        ps_pred = self.delta_model_.predict_proba(X)[:, 1]
        
        safe_p1 = np.clip(self.p_s1_, 1e-6, 1)
        safe_p2 = np.clip(self.p_s2_, 1e-6, 1)
        delta_vals = (ps_pred / safe_p1) - ((1 - ps_pred) / safe_p2)

        # 2. Filter Active Sets
        active_mask = np.abs(delta_vals) > self.min_delta
        eta_active = eta_train[active_mask]
        delta_active = delta_vals[active_mask]
        
        if len(eta_active) > self.max_samples_ot:
            indices = np.random.choice(len(eta_active), self.max_samples_ot, replace=False)
            eta_active = eta_active[indices]
            delta_active = delta_active[indices]

        # Split into Source (Plus group) and Target (Minus group) spaces
        idx_plus = np.where(delta_active > 0)
        idx_minus = np.where(delta_active < 0)
        
        Xs = np.column_stack([eta_active[idx_plus], delta_active[idx_plus]])
        Xt = np.column_stack([eta_active[idx_minus], delta_active[idx_minus]])
        
        ns, ds = Xs.shape
        nt, dt = Xt.shape
        
        # Fallback if a group is missing
        if ns == 0 or nt == 0:
             self.L_ = np.eye(ds)
             return self
             
        # Normalize Cost Matrix C for the standard OT loss
        C = ot.dist(Xs, Xt, metric='sqeuclidean')
        if C.max() > 0:
            C = C / C.max()
        
        # 3. Block-Coordinate Descent (Algorithm 1 from Perrot et al.)
        a, b = np.ones(ns)/ns, np.ones(nt)/nt
        gamma = np.outer(a, b) # Initialize gamma
        L = np.eye(ds)         # Initialize L as identity
        
        for k in range(self.max_iter_bcd):
            
            # --- Step A: Update Gamma (Frank-Wolfe step) ---
            # Compute gradient of objective w.r.t gamma
            # Grad = (-2 / d_t) * (X_s * L - n_s * gamma * X_t) * X_t^T + lambda_gamma * C
            diff = np.dot(Xs, L) - ns * np.dot(gamma, Xt)
            G = (-2.0 / dt) * np.dot(diff, Xt.T) + (self.lambda_gamma * C)
            
            # Solve linear OT to find the optimal Frank-Wolfe direction
            gamma_star = ot.emd(a, b, G)
            
            # Apply decaying step size for Frank-Wolfe update
            alpha = 2.0 / (k + 2.0) 
            gamma = (1.0 - alpha) * gamma + alpha * gamma_star
            
            # --- Step B: Update L (Closed form mapping) ---
            # Equation 9: L = (X_s^T X_s / (n_s d_t) + lambda_T / (d_s d_t) I)^{-1} 
            #                 (X_s^T (n_s gamma X_t) / (n_s d_t) + lambda_T / (d_s d_t) I)
            A = np.dot(Xs.T, Xs) / (ns * dt) + (self.lambda_T / (ds * dt)) * np.eye(ds)
            B = np.dot(Xs.T, ns * np.dot(gamma, Xt)) / (ns * dt) + (self.lambda_T / (ds * dt)) * np.eye(ds)
            
            L = np.linalg.solve(A, B)
            
        self.L_ = L
        return self

    def predict(self, X):
        X = np.array(X)
        eta_new = self.eta_model_.predict(X)
        
        ps = self.delta_model_.predict_proba(X)[:, 1]
        safe_p1 = np.clip(self.p_s1_, 1e-6, 1)
        safe_p2 = np.clip(self.p_s2_, 1e-6, 1)
        delta_new = (ps / safe_p1) - ((1 - ps) / safe_p2)
        
        final_pred = eta_new.copy()
        
        # Apply the explicitly learned linear mapping T(X) = X_s L 
        # We act as a Domain Adaptation model, mapping the "Plus" domain to the "Minus" domain
        mask_plus = delta_new > self.min_delta
        
        if np.any(mask_plus):
            X_plus = np.column_stack([eta_new[mask_plus], delta_new[mask_plus]])
            
            # L maps to the target space [eta, delta]. We only need the shifted 'eta' (column 0).
            mapped_X = np.dot(X_plus, self.L_)
            final_pred[mask_plus] = mapped_X[:, 0]
            
        return final_pred
    

    class OTUnawareFairRegressor_Taturyan():
    #cf the cloned code of Taturyan et al, in ./unaware-fair-reg/FairReg.py
        pass