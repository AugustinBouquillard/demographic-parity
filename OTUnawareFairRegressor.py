import numpy as np
import ot 
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp

class OTUnawareFairRegressor(BaseEstimator, RegressorMixin):
    """
    Implements Fair Regression via Optimal Transport and k-NN Interpolation.
    
    1. Solves the OT problem to find fair targets (barycenters) for training data.
    2. Learns the 'fairness shift' (correction) using k-NN on the (eta, delta) space.
    3. Predicts by applying this learned shift to the old regressor's output.
    """
    def __init__(self, base_regressor=None, base_classifier=None, n_neighbors=5):
        self.base_regressor = base_regressor if base_regressor else LinearRegression()
        self.base_classifier = base_classifier if base_classifier else LogisticRegression(solver='liblinear')
        
        # We use k-NN to regress the "shift" (correction)
        self.knn_correction_ = KNeighborsRegressor(n_neighbors=n_neighbors)
        self.scaler_ = StandardScaler()
        
        self.eta_model_ = None
        self.delta_model_ = None
        self.p_s1_ = None
        self.p_s0_ = None

    def fit(self, X, y, s):
        X = np.array(X)
        y = np.array(y)
        s = np.array(s)

        # --- 1. Fit Nuisance Models ---
        self.eta_model_ = clone(self.base_regressor).fit(X, y)
        eta_train = self.eta_model_.predict(X)

        self.p_s1_ = np.mean(s == 1)
        self.p_s0_ = np.mean(s == 0)
        self.delta_model_ = clone(self.base_classifier).fit(X, s)
        ps_pred = self.delta_model_.predict_proba(X)[:, 1]
        
        # Clip to avoid division by zero
        ps_pred = np.clip(ps_pred, 1e-6, 1 - 1e-6)
        delta_vals = (ps_pred / self.p_s1_) - ((1 - ps_pred) / self.p_s0_)

        # --- 2. Split Data by Delta ---
        # Group +: Delta > 0 (Advantaged)
        # Group -: Delta < 0 (Disadvantaged)
        eps = 1e-9
        idx_plus = np.where(delta_vals > eps)[0]
        idx_minus = np.where(delta_vals < -eps)[0]
        
        h_plus = eta_train[idx_plus]
        d_plus = np.abs(delta_vals[idx_plus])
        h_minus = eta_train[idx_minus]
        d_minus = np.abs(delta_vals[idx_minus])
        
        n_plus = len(h_plus)
        n_minus = len(h_minus)

        # --- 3. Compute Cost Matrix for OT ---
        # Formula: C(x1, x2) = (h1 - h2)^2 / (|d1| + |d2|)
        # Shape: (n_plus, n_minus)
        numer = (h_plus[:, None] - h_minus[None, :]) ** 2
        denom = (d_plus[:, None] + d_minus[None, :])
        M = numer / denom

        # --- 4. Solve Optimal Transport (Earth Mover's Distance) ---
        a = np.ones(n_plus) / n_plus
        b = np.ones(n_minus) / n_minus
        
        # Returns the transport matrix gamma (shape: n_plus x n_minus)
        gamma = ot.emd(a, b, M)

        # --- 5. Compute Fair Barycenters ---
        # For every pair (i, j), the optimal fair target is:
        # y* = (h_i/d_i + h_j/d_j) / (1/d_i + 1/d_j)
        
        inv_d_plus = 1.0 / d_plus
        inv_d_minus = 1.0 / d_minus
        
        # Compute pairwise optimal targets
        num_matrix = (h_plus * inv_d_plus)[:, None] + (h_minus * inv_d_minus)[None, :]
        den_matrix = inv_d_plus[:, None] + inv_d_minus[None, :]
        Y_opt_pairs = num_matrix / den_matrix
        
        # Aggregate targets for training points
        # For point i in Plus: Expected target = sum_j (gamma_ij / a_i) * Y_opt_ij
        y_fair_plus = np.sum(gamma * Y_opt_pairs, axis=1) * n_plus
        
        # For point j in Minus: Expected target = sum_i (gamma_ij / b_j) * Y_opt_ij
        y_fair_minus = np.sum(gamma * Y_opt_pairs, axis=0) * n_minus
        
        # Construct full training arrays
        # We train on the sufficient statistics (eta, delta)
        X_train_features = np.concatenate([
            np.column_stack((h_plus, delta_vals[idx_plus])),
            np.column_stack((h_minus, delta_vals[idx_minus]))
        ])
        
        # We learn the SHIFT: Correction = y_fair - y_original
        # This keeps the "old regressor" as the base.
        y_fair_shifts = np.concatenate([
            y_fair_plus - h_plus,
            y_fair_minus - h_minus
        ])

        # --- 6. Fit k-NN on the Shifts ---
        # Scale features for k-NN
        X_train_scaled = self.scaler_.fit_transform(X_train_features)
        self.knn_correction_.fit(X_train_scaled, y_fair_shifts)
        
        return self

    def predict(self, X):
        X = np.array(X)
        
        # 1. Get Old Regressor Prediction
        eta_new = self.eta_model_.predict(X)
        
        # 2. Get Delta
        ps = self.delta_model_.predict_proba(X)[:, 1]
        delta_new = (ps / self.p_s1_) - ((1 - ps) / self.p_s0_)
        
        # 3. Predict Correction via k-NN
        features_new = np.column_stack((eta_new, delta_new))
        features_scaled = self.scaler_.transform(features_new)
        
        pred_shifts = self.knn_correction_.predict(features_scaled)
        
        # 4. Apply Correction
        return eta_new + pred_shifts

