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

# print("fair unaware ot (gp + knn): ")
# evaluation_cross_validation(10, ot_reg, X, y, s )
# %%
# %%
# gamma with silverman rule
h = np.std(y)*1000**(-0.2)*1.06
print(h)
# %%

import numpy as np
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import scipy.stats as stats
from OTAwareFairRegressor import OTAwareFairRegressor
from sklearn.kernel_ridge import KernelRidge

noise_scale = 0.3
X, y, s = generate_linear_data(n = 2000, alpha_0 = 2, alpha_1 = 1, p = 0.5, noise_scale= noise_scale)
kernel = 2 * RBF(length_scale=3.0, length_scale_bounds=(1e-2, 1e2))
kernel_krr = KernelRidge(kernel='rbf', alpha=0.1, gamma = 0.3)

gp_reg = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer=10, alpha=2*noise_scale**2)




# print("unfair gp regressor: ")
# evaluation_cross_validation(5, gp_reg, X, y, s, prediction = "unfair")

# fair_derived_from_aware_model = OTAwareFairRegressor(base_estimator_model = gp_reg)

# print("fair aware ot (gp): ")
# evaluation_cross_validation(5, fair_derived_from_aware_model , X, y, s , prediction="aware")

unaware_model =   OTUnawareFairRegressor(base_regressor= gp_reg, n_neighbors= 1, kernel_krr= kernel_krr )

# print("fair unaware ot (gp+knn): ")
# evaluation_cross_validation(5, unaware_model , X, y, s, prediction = "knn" )

# print("fair unaware ot (gp+krr): ")
# evaluation_cross_validation(5, unaware_model , X, y, s, prediction = "krr" )

# %%


alpha_list = np.linspace(0.3, 5, 6)
alpha_len = len(alpha_list)
results_means = np.zeros((alpha_len  , 3))
results_stds =  np.zeros((alpha_len  , 3))
results_means_aware = np.zeros((alpha_len  , 3))
results_stds_aware =  np.zeros((alpha_len  , 3))
results_means_unfair = np.zeros((alpha_len  , 3))
results_stds_unfair =  np.zeros((alpha_len  , 3))
noise_scale = 0.3

# %%
for idx, alpha in enumerate(alpha_list ): 


    X, y, s = generate_linear_data(n = 1500, alpha_0 = alpha, alpha_1 = 1, p = 0.5, noise_scale= noise_scale)


    kernel = 2 * RBF(length_scale=1*alpha+1, length_scale_bounds=(1e-2, 1e2))
  

    gp_reg = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer=10, alpha=2*noise_scale**2)
 
    unaware_model =   OTUnawareFairRegressor(base_regressor= gp_reg, n_neighbors= 5)
    
    means, stds = evaluation_cross_validation(5, unaware_model , X, y, s, prediction = "knn" )
    results_means[idx] = means 
    results_stds[idx] = stds


for idx, alpha in enumerate(alpha_list ): 
    X, y, s = generate_linear_data(n = 2000, alpha_0 = alpha, alpha_1 = 1, p = 0.5, noise_scale= noise_scale)

    kernel = 2 * RBF(length_scale=1*alpha+1, length_scale_bounds=(1e-2, 1e2))

    gp_reg = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer=10, alpha=2*noise_scale**2)
    fair_derived_from_aware_model = OTAwareFairRegressor(base_estimator_model = gp_reg) 


    means, stds = evaluation_cross_validation(5, fair_derived_from_aware_model , X, y, s , prediction="aware")

    results_means_aware[idx] = means 
    results_stds_aware[idx] = stds 

# %%
for idx, alpha in enumerate(alpha_list ): 
    X, y, s = generate_linear_data(n = 2000, alpha_0 = alpha, alpha_1 = 1, p = 0.5, noise_scale= noise_scale)

    kernel = 2 * RBF(length_scale=1*alpha+1, length_scale_bounds=(1e-2, 1e2))

    gp_reg = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer=10, alpha=2*noise_scale**2)
   


    means, stds = evaluation_cross_validation(5, gp_reg , X, y, s , prediction="unfair")

    results_means_unfair[idx] = means 
    results_stds_unfair[idx] = stds 

# %%
alpha

# %%
results_means_aware_plug = np.zeros((alpha_len  , 3))
results_stds_aware_plug =  np.zeros((alpha_len  , 3))

for idx, alpha in enumerate(alpha_list ): 
    X, y, s = generate_linear_data(n = 2000, alpha_0 = alpha, alpha_1 = 1, p = 0.5, noise_scale= noise_scale)

    kernel = 2 * RBF(length_scale=1*alpha+1, length_scale_bounds=(1e-2, 1e2))

    gp_reg = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer=10, alpha=2*noise_scale**2)
    fair_derived_from_aware_model = OTAwareFairRegressor(base_estimator_model = gp_reg) 

    means, stds = evaluation_cross_validation(5, fair_derived_from_aware_model , X, y, s , prediction="plugin")

    results_means_aware_plug[idx] = means 
    results_stds_aware_plug[idx] = stds 
# %%
alpha_list
# %%
plt.scatter(X, y, c = s)
plt.plot()
# %%

# %%
alpha_list
# %%
indicators = ['MSE', 'Wasserstein 2', 'KS Distance']
colors = {'aware': '#1f77b4', 'unaware': '#ff7f0e', 'unfair': "#867AEC", 'aware_derived': "#4c7e15"}  # Blue and Orange

# --- 2. Plotting ---
fig, axes = plt.subplots(1, 3, figsize=(10, 3), sharex=True)

for i, ax in enumerate(axes):
    indicator_name = indicators[i]
    
        # Unfair Case
    ax.plot(alpha_list, results_means_unfair[:, i], 
            label='Unfair', color=colors['unfair'], lw=2, marker='s', markersize=4)
    ax.fill_between(alpha_list, 
                    results_means_unfair[:, i] - results_stds_unfair[:, i], 
                    results_means_unfair[:, i] + results_stds_unfair[:, i], 
                    color=colors['unfair'], alpha=0.15)
    
    # Aware Case
    ax.plot(alpha_list, results_means_aware[:, i], 
            label='Aware', color=colors['aware'], lw=2, marker='o', markersize=4)
    ax.fill_between(alpha_list, 
                    results_means_aware[:, i] - results_stds_aware[:, i], 
                    results_means_aware[:, i] + results_stds_aware[:, i], 
                    color=colors['aware'], alpha=0.15)
    

    ax.plot(alpha_list, results_means_aware_plug[:, i], 
            label='Aware derived', color=colors['aware_derived'], lw=2, marker='s', markersize=4)
    ax.fill_between(alpha_list, 
                    results_means_aware_plug[:, i] - results_stds_aware_plug[:, i], 
                    results_means_aware_plug[:, i] + results_stds_aware_plug[:, i], 
                    color=colors['aware_derived'], alpha=0.15)
    
    # Unaware Case
    ax.plot(alpha_list, results_means[:, i], 
            label='Unaware opt', color=colors['unaware'], lw=2, marker='s', markersize=4)
    ax.fill_between(alpha_list, 
                    results_means[:, i] - results_stds[:, i], 
                    results_means[:, i] + results_stds[:, i], 
                    color=colors['unaware'], alpha=0.15)
    



    # Formatting each subplot
    ax.set_title(f'{indicator_name}', fontsize=14)
    if i == 1 :
        ax.set_xlabel(r'discriminability $\alpha_0$', fontsize=12)
    # ax.set_ylabel('Value', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(alpha_list)
    # Legend only on the first or last plot to save space
    if i == 0:
        ax.legend(loc='best')

plt.tight_layout()

plt.show()
# %%

# visualisation
cmap = plt.get_cmap('tab10')
color_maj = cmap(0)  # Color for S=1 (Orange)
color_min = cmap(1)  # Color for S=2 (Green)
color_all = 'black'  # Color for the unfair regressor

plt.figure(figsize=(10, 6))

# Plot Data Points (Split by group for the legend)
plt.scatter(X[s == 1], y[s == 1], color=color_maj, alpha=0.5, s=30, 
            label='Data S=1 (Majority)')

plt.scatter(X[s == 2], y[s == 2], color=color_min, alpha=0.5, s=30, 
            label='Data S=2 (Minority)')

# Plot Regression Lines 
# Create X range for smooth lines
x_range_min = X.min() - 0.2
x_range_max = X.max() + 0.2
X_plot = np.linspace(x_range_min, x_range_max, 1000).reshape(-1, 1)

# Line for S=1


# Line for Unfair (Combined)
plt.plot(X_plot, fair_derived_from_aware_model.predict(X_plot), color=color_all, linestyle='--', 
         linewidth=2, label='Unfair Regressor (Combined)')

plt.title(" Bias in Generated Data (Gaussian process regressor)", fontsize=14)
plt.xlabel("Feature X")
plt.ylabel("Target Y")

# Legend
plt.legend(frameon=True, loc='best')

plt.tight_layout()
plt.show()
# %%
