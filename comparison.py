# Evaluation of the performance and the fairness for different methods

# methods include :
# unfair GP regressor, aware(GP), unaware(GP + kNN), unaware(GP + krr), aware derived (with S predicted instead of true S)

# unaware(GP + KNN) is kept as aware derived is kind of kNN with n = 2

# alpha_0 = 2 fix 

# For performance: MSE
# For fairness : Wasserstein-1/2, KS (maximum difference between the CFD)


# %%
import numpy as np 
import ot
from scipy.stats import ks_2samp
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
from OTAwareFairRegressor import OTAwareFairRegressor
from sklearn.kernel_ridge import KernelRidge

from OTUnawareFairRegressor import OTUnawareFairRegressor 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import sys
import os
# %%
def evaluation(y_unfair, y_fair, s_attr, p = 1):
    """
    Evaluate mse, wass_p, ks distance.
    y is always 1D.
    Parameters:
    s_attr : S = 1 (majority) or 2 (minority) 
    """
    
    mse = mean_squared_error(y_fair , y_unfair)
    y_fair_1 = y_fair[s_attr == 1]
    y_fair_2 = y_fair[s_attr == 2]
    a1 = np.ones_like(y_fair_1)/len(y_fair_1)
    a2 = np.ones_like(y_fair_2)/len(y_fair_2)
    wass_dist = ot.wasserstein_1d(y_fair_1, y_fair_2,a1, a2, p = p)
    ks_dist = ks_2samp(y_fair_1, y_fair_2).statistic

    return mse, wass_dist, ks_dist

def evaluation_cross_validation(k, model, X, y, s , prediction = None, p = 1):
    """
    Cross valisation on a dataset. 
    """
    y = y.reshape(-1, 1)
   
    spliter = KFold(n_splits=k, shuffle=True, random_state=42)

    fold_mse = np.zeros(k)
    fold_wass_dist = np.zeros(k)
    fold_ks_dist = np.zeros(k)
    
    for fold, (train_index, test_index) in enumerate(spliter.split(X)):
    
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        s_train, s_test = s[train_index], s[test_index]

        if prediction == "unfair":
            model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train, s_train)
        if prediction == "aware":
            y_pred = model.predict(X_test, s_test)
        elif prediction == "unfair" or "plug_in":
            y_pred = model.predict(X_test)

        else : 
            y_pred = model.predict(X_test, prediction = prediction)
        mse, wass, ks = evaluation(y_test, y_pred, s_test, p =p) 
        
        fold_mse[fold] = mse 
        fold_wass_dist[fold] = wass 
        fold_ks_dist[fold] = ks
    
    means = [np.mean(fold_mse), np.mean(fold_wass_dist), np.mean(fold_ks_dist)]
    stds = [np.std(fold_mse), np.std(fold_wass_dist), np.std(fold_ks_dist)] 
    formatted_means = [f"{m:.4f}" for m in means]
    formatted_stds = [f"{s:.4f}" for s in stds]

    print(f"Means: {formatted_means}")
    print(f"Stds:  {formatted_stds}")
    return means, stds


# %%

def generate_linear_data(n , alpha_0, alpha_1, p = 0.3, x_scale = 1, noise_scale = 1, seed = 42):
    """
    Generate 1D linear data.  
    (X = normal(0, x_scale) - alpha_0 * S, 
    Y = X - alpha_1 * S + normal(0, noise_scale, n) )
    S = 1 for majority
    S = 2 for minority

    :param n: size of dataset
    :param alpha_0: X = normal(0, x_scale) - alpha_0*S 
    :param alpha_1: Y = X - alpha_1*S 
    :param p: probability for S = 2 (parameter in random.binomial)
    :param x_scale: X = normal(0, x_scale) - alpha_0*S 
    :param seed: random seed
    """
    np.random.seed(seed)
    S = np.random.binomial(1, p, n)+1 # Binary sensitive attributes

    X = np.random.normal(0, x_scale, n) - alpha_0 * S
    X = X.reshape(-1, 1)

    # Y depends on X and S 
    Y = - alpha_1*S + 0.5 * X.flatten() + np.random.normal(0, noise_scale, n)
   
    return X, Y, S


# %%


# Cross-validation for different methods (mse, wass_1, ks)


noise_scale = 0.3
X, y, s = generate_linear_data(n = 2000, alpha_0 = 2, alpha_1 = 1, p = 0.5, noise_scale= noise_scale)

kernel = 2 * RBF(length_scale=3.0, length_scale_bounds=(1e-2, 1e2))

# gamma with silverman rule \approx 0.3 for krr
h = np.std(y)*1000**(-0.2)*1.06 
kernel_krr = KernelRidge(kernel='rbf', alpha=0.1, gamma = 0.3)

gp_reg = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer=10, alpha=2*noise_scale**2)

print("unfair gp regressor: ")
evaluation_cross_validation(5, gp_reg, X, y, s, prediction = "unfair")

aware_model = OTAwareFairRegressor(base_estimator_model = gp_reg)

print("fair aware (gp): ")
evaluation_cross_validation(5, aware_model , X, y, s , prediction="aware")

unaware_model =   OTUnawareFairRegressor(base_regressor= gp_reg, n_neighbors= 2)

print("fair unaware (gp+knn): ")
evaluation_cross_validation(5, unaware_model , X, y, s, prediction = "knn" )

unaware_krr_model =   OTUnawareFairRegressor(base_regressor= gp_reg, n_neighbors= 1, kernel_krr= kernel_krr )
print("fair unaware (gp+krr): ")
evaluation_cross_validation(5, unaware_krr_model , X, y, s, prediction = "krr" )

aware_derived_model = OTAwareFairRegressor(base_estimator_model = gp_reg)
print("fair aware derived (gp): ")
evaluation_cross_validation(5, aware_derived_model , X, y, s, prediction = "plugin" )


current_dir = os.getcwd()
#print(f"Notebook is running in: {current_dir}")

folder_path = os.path.abspath(os.path.join(current_dir, 'unaware-fair-reg-3rd-method'))
#print(f"Looking for module folder at: {folder_path}")
#print(f"Does this folder exist? {os.path.exists(folder_path)}")

if folder_path not in sys.path:
    sys.path.insert(0, folder_path)
from FairReg import FairReg


def cross_validation_taturyan(k, X, y, s , p = 1):
    """
    Cross valisation on a dataset. For unaware regressor Taturyan.
    """
    y = y.reshape(-1, 1)
   
    spliter = KFold(n_splits=k, shuffle=True, random_state=42)

    fold_mse = np.zeros(k)
    fold_wass_dist = np.zeros(k)
    fold_ks_dist = np.zeros(k)
    
    for fold, (train_index, test_index) in enumerate(spliter.split(X)):
    
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = y[train_index], y[test_index]
        s_train, s_test = s[train_index], s[test_index]

        kernel = 2 * RBF(length_scale=3.0, length_scale_bounds=(1e-2, 1e2))
        
        gp_reg = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer=10, alpha=2*noise_scale**2).fit(X_train, y_train)
        

        proxy_classifier = LogisticRegression()
        proxy_classifier.fit(X_train, s_train)

        
        B_val = np.max(np.abs(y_train)) 

        # K: Number of sensitive attribute groups
        unique_groups = np.unique(s_train)
        K_val = len(unique_groups)

        # p: Frequencies of each sensitive group in the training data
        p_val = [np.mean(s_train == s) for s in unique_groups]

        # eps: Epsilon thresholds for demographic parity (tolerance for unfairness)
        eps_val = [0.00001 for _ in range(K_val)] 

        # T: Number of iterations for the stochastic gradient descent
        T_val = 1000000

        # 3. Initialize the FairReg model
        fair_reg_taturyan = FairReg(
            base_method=gp_reg,
            classifier=proxy_classifier,
            B=B_val,
            K=K_val,
            p=p_val,
            eps=eps_val,
            T=T_val
        )

        # 4. Fit the fairness weights (w_est) using X_train
        fair_reg_taturyan.fit(X_train)

        # 5. Predict on the test set
        y_pred_taturyan = fair_reg_taturyan.predict(X_test)


        mse, wass, ks = evaluation(y_test, y_pred_taturyan, s_test, p = p) 
        
        fold_mse[fold] = mse 
        fold_wass_dist[fold] = wass 
        fold_ks_dist[fold] = ks
    
    means = [np.mean(fold_mse), np.mean(fold_wass_dist), np.mean(fold_ks_dist)]
    stds = [np.std(fold_mse), np.std(fold_wass_dist), np.std(fold_ks_dist)] 
    formatted_means = [f"{m:.4f}" for m in means]
    formatted_stds = [f"{s:.4f}" for s in stds]

    print(f"Means: {formatted_means}")
    print(f"Stds:  {formatted_stds}")
    return means, stds


print("fair unaware taturyan: ")
cross_validation_taturyan(5,X, y, s )

# %%

# crosse validation (mse, wass_2, ks)

noise_scale = 0.3
X, y, s = generate_linear_data(n = 2000, alpha_0 = 2, alpha_1 = 1, p = 0.5, noise_scale= noise_scale)

kernel = 2 * RBF(length_scale=3.0, length_scale_bounds=(1e-2, 1e2))
kernel_krr = KernelRidge(kernel='rbf', alpha=0.1, gamma = 0.3)

gp_reg = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer=10, alpha=2*noise_scale**2)

print("unfair gp regressor: ")
evaluation_cross_validation(5, gp_reg, X, y, s, prediction = "unfair", p = 2)

aware_model = OTAwareFairRegressor(base_estimator_model = gp_reg)

print("fair aware (gp): ")
evaluation_cross_validation(5, aware_model , X, y, s , prediction="aware", p = 2)

unaware_model =   OTUnawareFairRegressor(base_regressor= gp_reg, n_neighbors= 2)

print("fair unaware (gp+knn): ")
evaluation_cross_validation(5, unaware_model , X, y, s, prediction = "knn" , p = 2)

unaware_krr_model =   OTUnawareFairRegressor(base_regressor= gp_reg, n_neighbors= 1, kernel_krr= kernel_krr )
print("fair unaware (gp+krr): ")
evaluation_cross_validation(5, unaware_krr_model , X, y, s, prediction = "krr", p = 2 )

aware_derived_model = OTAwareFairRegressor(base_estimator_model = gp_reg)
print("fair aware derived (gp): ")
evaluation_cross_validation(5, aware_derived_model , X, y, s, prediction = "plugin", p = 2)

print("unaware taturyan (gp): ")
# W2 tatyuryan 
cross_validation_taturyan(5,X, y, s , p = 2)


