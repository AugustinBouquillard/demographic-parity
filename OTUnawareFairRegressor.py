import numpy as np
import ot 
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor

class OTUnawareFairRegressor(BaseEstimator, RegressorMixin):
    """
    Implements Fair Regression via optimal transport 
    (find the barycenter and estimate the transport plan).
    """
    def __init__(self, base_regressor=None, base_classifier=None, n_neighbors=5, kernel_krr=KernelRidge(kernel='rbf', alpha=0.1, gamma=0.3), random_forest=RandomForestRegressor(max_depth=2)):
        # FIXED: Explicitly use "is not None" to avoid calling __len__ on unfitted scikit-learn models
        self.base_regressor = base_regressor if base_regressor is not None else LinearRegression()
        
        # Standard Logistic Regression (NO class_weight='balanced' to preserve true probabilities)
        self.base_classifier = base_classifier if base_classifier is not None else LogisticRegression(solver='liblinear')
        
        self.knn_plus_ = KNeighborsRegressor(n_neighbors=n_neighbors)
        self.knn_minus_ = KNeighborsRegressor(n_neighbors=n_neighbors) 
        self.linear_mapping_plus_ = ot.da.LinearGWTransport()
        self.linear_mapping_minus_ = ot.da.LinearGWTransport() 
        self.scaler_ = StandardScaler()
        self.krr_plus_ = clone(kernel_krr)
        self.krr_minus_ = clone(kernel_krr)

        self.random_forest_plus_ = clone(random_forest)
        self.random_forest_minus_ = clone(random_forest)

        self.eta_model_ = None
        self.delta_model_ = None
        self.s1_ = None
        self.s2_ = None
        self.p_s1_ = None
        self.p_s2_ = None
        
        self.y_fair_plus = None
        self.y_fair_minus = None
        self.h_plus = None
        self.h_minus = None 
        self.delta_predict = None

    def fit(self, X, y, s):
        X = np.array(X)
        y = np.array(y)
        s = np.array(s).flatten()

        # 1. Dynamically identify the two sensitive groups
        classes = np.sort(np.unique(s))
        if len(classes) != 2:
            raise ValueError(f"Expected exactly 2 sensitive groups, found {len(classes)}.")
        self.s1_, self.s2_ = classes[0], classes[1]

        # 2. Fit Bayesian Models
        self.eta_model_ = clone(self.base_regressor).fit(X, y)
        eta_train = self.eta_model_.predict(X)

        # 3. Calculate Empirical Frequencies
        self.p_s1_ = np.clip(np.mean(s == self.s1_), a_min=1e-6, a_max=1-1e-6)
        self.p_s2_ = np.clip(np.mean(s == self.s2_), a_min=1e-6, a_max=1-1e-6)
        
        self.delta_model_ = clone(self.base_classifier).fit(X, s)
        
        # Extract P(S=s1 | X). Index 0 aligns with classes_[0] which is self.s1_
        ps_pred = self.delta_model_.predict_proba(X)[:, 0]
        
        # Calculate Delta
        delta_vals = (ps_pred / self.p_s1_) - ((1 - ps_pred) / self.p_s2_)

        # 4. Split Data by Delta
        eps = 1e-9
        idx_plus = np.where(delta_vals > eps)[0]
        idx_minus = np.where(delta_vals < -eps)[0]
        
        # Fail-safe against total proxy collapse
        if len(idx_plus) == 0 or len(idx_minus) == 0:
            raise ValueError("Proxy collapsed: Features contain zero signal about the sensitive attribute. OT mapping cannot be applied.")
        
        h1 = eta_train[idx_plus]
        h2 = eta_train[idx_minus]
        
        n1 = len(h1)
        n2 = len(h2)

        # 5. Cost Matrix for OT
        d1 = np.abs(delta_vals[idx_plus])
        d2 = np.abs(delta_vals[idx_minus])
        numer = (h1[:, None] - h2[None, :]) ** 2
        denom = (d1[:, None] + d2[None, :])
        M = numer / denom 

        # 6. Solve Optimal Transport
        a = np.ones(n1) / n1
        b = np.ones(n2) / n2
        gamma = ot.emd(a, b, M)

        # 7. Recover the fair barycenter
        inv_d1 = 1.0 / d1
        inv_d2 = 1.0 / d2
        num_matrix = (h1 * inv_d1)[:, None] + (h2 * inv_d2)[None, :]
        den_matrix = inv_d1[:, None] + inv_d2[None, :]
        Y_opt_pairs = num_matrix / den_matrix
        
        y_fair_plus = np.sum(gamma * Y_opt_pairs, axis=1) * n1
        y_fair_minus = np.sum(gamma * Y_opt_pairs, axis=0) * n2
        
        # Construct full training arrays
        X_train_features = np.concatenate([
            np.column_stack((h1, delta_vals[idx_plus])),
            np.column_stack((h2, delta_vals[idx_minus]))
        ])
        y_fair = np.concatenate([y_fair_plus, y_fair_minus])

        self.y_fair_plus = y_fair_plus.ravel()
        self.y_fair_minus = y_fair_minus.ravel()
        self.h_plus = h1.reshape(-1, 1)
        self.h_minus = h2.reshape(-1, 1)

        # 8. Fit Mappings
       
        self.knn_plus_.fit(self.h_plus, self.y_fair_plus)
        self.knn_minus_.fit(self.h_minus, self.y_fair_minus)

        self.krr_plus_.fit(self.h_plus, self.y_fair_plus)
        self.krr_minus_.fit(self.h_minus, self.y_fair_minus)
  
        self.random_forest_plus_.fit(self.h_plus, self.y_fair_plus)
        self.random_forest_minus_.fit(self.h_minus, self.y_fair_minus)

        self.linear_mapping_plus_.fit(Xs=self.h_plus, Xt=self.y_fair_plus.reshape(-1, 1))
        self.linear_mapping_minus_.fit(Xs=self.h_minus, Xt=self.y_fair_minus.reshape(-1, 1))

        return self

    def predict(self, X, prediction="knn"):
        X = np.array(X)

        eta_new = self.eta_model_.predict(X)
        
        # Extract P(S=s1 | X)
        ps = self.delta_model_.predict_proba(X)[:, 0]
        delta_new = (ps / self.p_s1_) - ((1 - ps) / self.p_s2_)
        self.delta_predict = delta_new

        pos_mask = delta_new >= 0
        neg_mask = ~pos_mask  
        
        if prediction == "linear":
            if eta_new.ndim == 1:
                eta_new = eta_new.reshape(-1, 1)
            pred_linear = np.zeros(len(X))
            if np.any(pos_mask):
                pred_linear[pos_mask] = self.linear_mapping_plus_.transform(eta_new[pos_mask].reshape(-1, 1)).ravel()

            if np.any(neg_mask):
                pred_linear[neg_mask] = self.linear_mapping_minus_.transform(eta_new[neg_mask].reshape(-1, 1)).ravel()

            return pred_linear
            
        elif prediction == "krr":
            pred_krr = np.zeros(len(X)) 
             
            if np.any(pos_mask):
                pred_krr[pos_mask] = self.krr_plus_.predict(eta_new[pos_mask].reshape(-1, 1))

            if np.any(neg_mask):
                pred_krr[neg_mask] = self.krr_minus_.predict(eta_new[neg_mask].reshape(-1, 1))

            return pred_krr
            
        elif prediction == "random_forest":
            pred_rf = np.zeros(len(X)) 
             
            if np.any(pos_mask):
                pred_rf[pos_mask] = self.random_forest_plus_.predict(eta_new[pos_mask].reshape(-1, 1))

            if np.any(neg_mask):
                pred_rf[neg_mask] = self.random_forest_minus_.predict(eta_new[neg_mask].reshape(-1, 1))

            return pred_rf
        else: 
            pred_knn = np.zeros(len(X)) 
             
            if np.any(pos_mask):
                pred_knn[pos_mask] = self.knn_plus_.predict(eta_new[pos_mask].reshape(-1, 1)).ravel()

            if np.any(neg_mask):
                pred_knn[neg_mask] = self.knn_minus_.predict(eta_new[neg_mask].reshape(-1, 1)).ravel()

            return pred_knn
            
        

