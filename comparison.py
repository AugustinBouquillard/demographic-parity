# Evaluation of the performance and the fairness 

# For performance: MSE
# For fairness : Wasserstein-2, KS (maximum difference between the CFD)
# %%
import numpy as np 
from sklearn.metrics import mean_squared_error
import ot
from scipy.stats import ks_2samp
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# %%
def evaluation(y_unfair, y_fair, s_attr):
    """
    y is always 1D.
    Parameters:
    s_attr : S = 1 (majority) or 2 (minority) 
    """
    
    mse = mean_squared_error(y_fair , y_unfair)
    y_fair_1 = y_fair[s_attr == 1]
    y_fair_2 = y_fair[s_attr == 2]
    a1 = np.ones_like(y_fair_1)/len(y_fair_1)
    a2 = np.ones_like(y_fair_2)/len(y_fair_2)
    wass_dist = np.sqrt(ot.wasserstein_1d(y_fair_1, y_fair_2,a1, a2))
    ks_dist = ks_2samp(y_fair_1, y_fair_2).statistic

    return mse, wass_dist, ks_dist

def evaluation_cross_validation(k, model, X, y, s , prediction = None):
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
    
        model.fit(X_train, y_train, s_train)
        
        y_pred = model.predict(X_test, prediction = prediction)
        mse, wass, ks = evaluation(y_test, y_pred, s_test) 
        
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

from OTUnawareFairRegressor import OTUnawareFairRegressor 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

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

noise_scale = 0.3
X, y, s = generate_linear_data(n = 1000, alpha_0 = 2, alpha_1 = 1, p = 0.5, noise_scale= noise_scale)
kernel = 2 * RBF(length_scale=3.0, length_scale_bounds=(1e-2, 1e2))
gp_reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=noise_scale**2)
ot_reg = OTUnawareFairRegressor(base_regressor= gp_reg)

print("fair unaware ot (gp + knn): ")
evaluation_cross_validation(10, ot_reg, X, y, s )
# %%


from sklearn.kernel_ridge import KernelRidge
for gamma in [1.2, 0.3, 0.4]:
    kernel_krr = KernelRidge(kernel='rbf', alpha=0.1, gamma= gamma)

    ot_reg_krr = OTUnawareFairRegressor(base_regressor= gp_reg, kernel_krr = kernel_krr)

    print(f"fair unaware ot (gp + krr) with gamma {gamma}: ")
    print(evaluation_cross_validation(5, ot_reg_krr, X, y, s, prediction= "krr" ))
# %%
# gamma with silverman rule
h = np.std(y)*1000**(-0.2)*1.06
print(h)
# %%
