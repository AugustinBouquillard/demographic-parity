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

    s = 1, majority, mu +  
    s = 2, minority, mu - 
    """
    def __init__(self, base_regressor=None, base_classifier=None, n_neighbors=5):
        self.base_regressor = base_regressor if base_regressor else LinearRegression() # fit_intercept is True by defaut
        self.base_classifier = base_classifier if base_classifier else LogisticRegression(solver='liblinear')
        
        
        self.knn_correction_ = KNeighborsRegressor(n_neighbors=n_neighbors)
        self.linear_mapping_plus = ot.da.LinearGWTransport()
        self.linear_mapping_minus = ot.da.LinearGWTransport() 
        self.scaler_ = StandardScaler()
        
        self.eta_model_ = None
        self.delta_model_ = None
        self.p_s1_ = None
        self.p_s2_ = None
        
        
        self.y_fair_plus = None
        self.y_fair_minus = None
        self.h_plus = None
        self.h_minus = None 

    def fit(self, X, y, s):
        X = np.array(X)
        y = np.array(y)
        s = np.array(s)

        # --- 1. Fit Bayesian Models ---
        self.eta_model_ = clone(self.base_regressor).fit(X, y)
        eta_train = self.eta_model_.predict(X)

        self.p_s1_ = np.mean(s == 1)
        self.p_s2_ = np.mean(s == 2)
        self.delta_model_ = clone(self.base_classifier).fit(X, s)
        ps_pred = self.delta_model_.predict_proba(X)[:, 1]
        
        # Clip to avoid division by zero
        ps_pred = np.clip(ps_pred, 1e-6, 1 - 1e-6)
        delta_vals = (ps_pred / self.p_s1_) - ((1 - ps_pred) / self.p_s2_)

        # --- 2. Split Data by Delta ---
        # Group +: Delta > 0 (Advantaged, s = 1)
        # Group -: Delta < 0 (Disadvantaged, s = 2)
        eps = 1e-9
        idx_plus = np.where(delta_vals > eps)[0]
        idx_minus = np.where(delta_vals < -eps)[0]
        
        h1 = eta_train[idx_plus]
        d1 = np.abs(delta_vals[idx_plus])
        h2 = eta_train[idx_minus]
        d2 = np.abs(delta_vals[idx_minus])
        
        n1 = len(h1)
        n2 = len(h2)

        # --- 3. Compute Cost Matrix for OT ---
        # Formula: C(x1, x2) = (h1 - h2)^2 / (|d1| + |d2|)
        # Shape: (n1, n2) by eq(13)
        numer = (h1[:, None] - h2[None, :]) ** 2
        denom = (d1[:, None] + d2[None, :])
        M = numer / denom 

        # --- 4. Solve Optimal Transport (Earth Mover's Distance) ---
        a = np.ones(n1) / n1
        b = np.ones(n2) / n2
 
        # Returns the transport matrix gamma (shape: n1 x n2)
        gamma = ot.emd(a, b, M)

        # --- 5. Compute Fair Barycenters ---
        # For every pair (i, j), the optimal fair target is:
        # y* = (h_i/d_i + h_j/d_j) / (1/d_i + 1/d_j)
        
        inv_d1 = 1.0 / d1
        inv_d2 = 1.0 / d2
        
        # Compute pairwise optimal targets eq(15)
        num_matrix = (h1 * inv_d1)[:, None] + (h2 * inv_d2)[None, :]
        den_matrix = inv_d1[:, None] + inv_d2[None, :]
        Y_opt_pairs = num_matrix / den_matrix
        
        # Aggregate targets for training points
        # For point i in Plus: Expected target = sum_j (gamma_ij / a_i) * Y_opt_ij
        y_fair_plus = np.sum(gamma * Y_opt_pairs, axis=1) * n1
        
        # For point j in Minus: Expected target = sum_i (gamma_ij / b_j) * Y_opt_ij
        y_fair_minus = np.sum(gamma * Y_opt_pairs, axis=0) * n2
        
        # Construct full training arrays
        # We train on the sufficient statistics (eta, delta)
        X_train_features = np.concatenate([
            np.column_stack((h1, delta_vals[idx_plus])),
            np.column_stack((h2, delta_vals[idx_minus]))
        ])

        h = np.concatenate([h1, h2])
        
        y_fair = np.concatenate([
            y_fair_plus ,
            y_fair_minus
        ])
        self.y_fair_plus = y_fair_plus.reshape(-1,1)
        self.y_fair_minus = y_fair_minus.reshape(-1,1)
        self.h_plus = h1.reshape(-1,1)
        self.h_minus = h2.reshape(-1,1)

        # --- Fit k-NN on the Shifts ---
        # Scale features for k-NN
        X_train_scaled = self.scaler_.fit_transform(X_train_features)
        self.knn_correction_.fit(X_train_scaled, y_fair)
        

        # -- Fit a linear mapping ---
        
        self.linear_mapping_plus.fit(Xs=self.h_plus, Xt=self.y_fair_plus)
        self.linear_mapping_minus.fit(Xs=self.h_minus, Xt= self.y_fair_minus)
        return self

    def predict(self, X, prediction = "linear"):
        X = np.array(X)

        # 1. Get Old Regressor Prediction
        eta_new = self.eta_model_.predict(X)
        
        # 2. Get Delta
        ps = self.delta_model_.predict_proba(X)[:, 1]
        delta_new = (ps / self.p_s1_) - ((1 - ps) / self.p_s2_)

        # 3. Predict via k-NN
        features_new = np.column_stack((eta_new, delta_new))
        features_scaled = self.scaler_.transform(features_new)
        
        pred_knn = self.knn_correction_.predict(features_scaled)
        
        pred_linear = np.zeros(len(X))
        for idx, delta in enumerate(delta_new):
            if delta >=0:
                pred_linear[idx] = self.linear_mapping_plus.transform(self.eta_model_.predict(X[idx].reshape(-1, 1)))[0][0]
            else: 
   
                pred_linear[idx] = self.linear_mapping_minus.transform(self.eta_model_.predict(X[idx].reshape(-1, 1)))[0][0]

        if prediction == "linear":
            return pred_linear
        else : 
            return  pred_knn

