# Evaluation of the impact of alpha_0

# methods include :
# unfair GP regressor, aware(GP), unaware(GP + kNN), aware derived (with S predicted instead of true S)

# For performance: MSE
# For fairness : Wasserstein-1, KS (maximum difference between the CFD)

# %%
import numpy as np 
from sklearn.metrics import mean_squared_error
import ot
from scipy.stats import ks_2samp
from sklearn.model_selection import KFold
from OTUnawareFairRegressor import OTUnawareFairRegressor 
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from OTAwareFairRegressor import OTAwareFairRegressor
from scipy.stats import wasserstein_distance

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
    wass_dist = ot.wasserstein_1d(y_fair_1, y_fair_2,a1, a2)
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

noise_scale = 0.3
X, y, s = generate_linear_data(n = 1000, alpha_0 = 2, alpha_1 = 1, p = 0.5, noise_scale= noise_scale)
kernel = 2 * RBF(length_scale=3.0, length_scale_bounds=(1e-2, 1e2))
gp_reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=noise_scale**2)
ot_reg = OTUnawareFairRegressor(base_regressor= gp_reg)


alpha_list = np.linspace(0.3, 4.5, 6)
alpha_len = len(alpha_list)
results_means = np.zeros((alpha_len  , 3))
results_stds =  np.zeros((alpha_len  , 3))
results_means_aware = np.zeros((alpha_len  , 3))
results_stds_aware =  np.zeros((alpha_len  , 3))
results_means_unfair = np.zeros((alpha_len  , 3))
results_stds_unfair =  np.zeros((alpha_len  , 3))
results_means_aware_plug = np.zeros((alpha_len  , 3))
results_stds_aware_plug =  np.zeros((alpha_len  , 3))
noise_scale = 0.3

# %%
# run cross validation to get the changes of metrics with different alphas
# unaware (gp + knn)
for idx, alpha in enumerate(alpha_list ): 


    X, y, s = generate_linear_data(n = 2000, alpha_0 = alpha, alpha_1 = 1, p = 0.5, noise_scale= noise_scale)


    kernel = 2 * RBF(length_scale=1*alpha+1, length_scale_bounds=(1e-2, 1e2))
  

    gp_reg = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer=10, alpha=2*noise_scale**2)
 
    unaware_model =   OTUnawareFairRegressor(base_regressor= gp_reg, n_neighbors= 5)
    
    means, stds = evaluation_cross_validation(5, unaware_model , X, y, s, prediction = "knn" )
    results_means[idx] = means 
    results_stds[idx] = stds

# aware 
for idx, alpha in enumerate(alpha_list ): 
    X, y, s = generate_linear_data(n = 2000, alpha_0 = alpha, alpha_1 = 1, p = 0.5, noise_scale= noise_scale)

    kernel = 2 * RBF(length_scale=1*alpha+1, length_scale_bounds=(1e-2, 1e2))

    gp_reg = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer=10, alpha=2*noise_scale**2)
    fair_derived_from_aware_model = OTAwareFairRegressor(base_estimator_model = gp_reg) 


    means, stds = evaluation_cross_validation(5, fair_derived_from_aware_model , X, y, s , prediction="aware")

    results_means_aware[idx] = means 
    results_stds_aware[idx] = stds 


# unfair (gp)
for idx, alpha in enumerate(alpha_list ): 
    X, y, s = generate_linear_data(n = 2000, alpha_0 = alpha, alpha_1 = 1, p = 0.5, noise_scale= noise_scale)

    kernel = 2 * RBF(length_scale=1*alpha+1, length_scale_bounds=(1e-2, 1e2))

    gp_reg = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer=10, alpha=2*noise_scale**2)
   


    means, stds = evaluation_cross_validation(5, gp_reg , X, y, s , prediction="unfair")

    results_means_unfair[idx] = means 
    results_stds_unfair[idx] = stds 


# aware (plug in)

for idx, alpha in enumerate(alpha_list ): 
    X, y, s = generate_linear_data(n = 2000, alpha_0 = alpha, alpha_1 = 1, p = 0.5, noise_scale= noise_scale)

    kernel = 2 * RBF(length_scale=1*alpha+1, length_scale_bounds=(1e-2, 1e2))

    gp_reg = GaussianProcessRegressor(kernel = kernel, n_restarts_optimizer=10, alpha=2*noise_scale**2)
    fair_derived_from_aware_model = OTAwareFairRegressor(base_estimator_model = gp_reg) 

    means, stds = evaluation_cross_validation(5, fair_derived_from_aware_model , X, y, s , prediction="plugin")

    results_means_aware_plug[idx] = means 
    results_stds_aware_plug[idx] = stds 


# %%
# final visualisation
indicators = ['MSE', 'Wasserstein 1', 'KS Distance']
colors = {'aware': '#1f77b4', 'unaware': '#ff7f0e', 'unfair': "#867AEC", 'aware_derived': "#4c7e15"}  # Blue and Orange

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
            label='Unaware', color=colors['unaware'], lw=2, marker='s', markersize=4)
    ax.fill_between(alpha_list, 
                    results_means[:, i] - results_stds[:, i], 
                    results_means[:, i] + results_stds[:, i], 
                    color=colors['unaware'], alpha=0.15)
    

    ax.set_title(f'{indicator_name}', fontsize=14)
    if i == 1 :
        ax.set_xlabel(r'discriminability $\alpha_0$', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6)
    plt.xticks(alpha_list)
    if i == 0:
        ax.legend(loc='best')

plt.tight_layout()
plt.show()

# %%
# histograms visualisation for some values of alpha
# base regressor eta: linear

n_points = 2000  # LOT of points for smooth histograms
alphas =  [0.15, 1.5, 3.0]  # From no separability to perfect separability
alphas_to_plot = alphas.copy()  # Specific alphas to visualize histograms for
n_runs = 1  # run experiment once just for histogram
noise_scale = 0.3
histogram_data = {}


mse_unfair_mean, mse_unfair_std = [], []
mse_fair_mean, mse_fair_std = [], []
w1_unfair_mean, w1_unfair_std = [], []
w1_fair_mean, w1_fair_std = [], []

print(f"Running experiment over alpha values with {n_runs} runs per alpha...")

# Loop through different alpha (separability) values
for alpha in alphas:
    temp_mse_unf, temp_mse_fair = [], []
    temp_w1_unf, temp_w1_fair = [], []
    temp_ks_unf, temp_ks_fair = [], [] 

    for run in range(n_runs):
        X_exp, Y_exp, S_exp = generate_linear_data(
            n=n_points, alpha_0=alpha, alpha_1=1, x_scale=1, noise_scale=noise_scale, seed=run + int(alpha*100)
        )
        
        X_train_exp, X_test_exp, Y_train_exp, Y_test_exp, S_train_exp, S_test_exp = train_test_split(
            X_exp, Y_exp, S_exp, train_size=0.8, random_state=run
        )
        
        # Train Unfair Regressor
        std_reg_exp = LinearRegression().fit(X_train_exp, Y_train_exp)
        y_unfair_exp = std_reg_exp.predict(X_test_exp)
        
        
        try:
            # Train OT Unaware Fair Regressor
            ot_reg_exp = OTUnawareFairRegressor()
            ot_reg_exp.fit(X_train_exp, Y_train_exp, S_train_exp)
            y_fair_exp = ot_reg_exp.predict(X_test_exp, prediction="knn")
            delta_exp = ot_reg_exp.delta_predict
            
        except AssertionError:
            # If proxy collapses because alpha is too low (no separability)
            if run == 0:
                print(f"Alpha {alpha:.2f}: Proxy collapsed (no separability). Using unfair baseline.")
            y_fair_exp = y_unfair_exp.copy()
            delta_exp = np.random.randn(len(y_fair_exp)) # Dummy delta
            
        # Split condition based on delta
        mask_pos = (S_test_exp == 1)
        mask_neg = (S_test_exp == 2)
        
        # Ensure we don't calculate Wasserstein on empty arrays
        if sum(mask_pos) > 0 and sum(mask_neg) > 0:
            w1_unf = wasserstein_distance(y_unfair_exp[mask_pos], y_unfair_exp[mask_neg])
            w1_f = wasserstein_distance(y_fair_exp[mask_pos], y_fair_exp[mask_neg])
            ks_unf = ks_2samp(y_unfair_exp[mask_pos], y_unfair_exp[mask_neg]).statistic 
            ks_f = ks_2samp(y_fair_exp[mask_pos], y_fair_exp[mask_neg]).statistic 
            
        else:
            w1_unf, w1_f = 0.0, 0.0
        
        temp_mse_unf.append(mean_squared_error(Y_test_exp, y_unfair_exp))
        temp_mse_fair.append(mean_squared_error(Y_test_exp, y_fair_exp))
        temp_w1_unf.append(w1_unf)
        temp_w1_fair.append(w1_f)
        temp_ks_unf.append(ks_unf)
        temp_ks_fair.append(ks_f)
        

        if run == 0 and any(np.isclose(alpha, a, atol=0.1) for a in alphas_to_plot) and len(histogram_data) < len(alphas_to_plot):
            histogram_data[alpha] = {
                'y_u': y_unfair_exp, 'y_f': y_fair_exp, 
                'mask_pos': mask_pos, 'mask_neg': mask_neg, 'w1_f': w1_f, 'ks_f': ks_f

            }
            
    # Calculate Mean and Standard Deviation for the current alpha
    mse_unfair_mean.append(np.mean(temp_mse_unf))
    mse_unfair_std.append(np.std(temp_mse_unf))
    
    mse_fair_mean.append(np.mean(temp_mse_fair))
    mse_fair_std.append(np.std(temp_mse_fair))
    
    w1_unfair_mean.append(np.mean(temp_w1_unf))
    w1_unfair_std.append(np.std(temp_w1_unf))
    
    w1_fair_mean.append(np.mean(temp_w1_fair))
    w1_fair_std.append(np.std(temp_w1_fair))


# %%
# plot

alphas = np.array(alphas)
mse_unfair_mean, mse_unfair_std = np.array(mse_unfair_mean), np.array(mse_unfair_std)
mse_fair_mean, mse_fair_std = np.array(mse_fair_mean), np.array(mse_fair_std)
w1_unfair_mean, w1_unfair_std = np.array(w1_unfair_mean), np.array(w1_unfair_std)
w1_fair_mean, w1_fair_std = np.array(w1_fair_mean), np.array(w1_fair_std)

print("Experiment complete. Plotting results...")

# Plot Smooth Histograms for specific Alphas
fig, axes = plt.subplots(len(histogram_data), 2, figsize=(10, 2 * len(histogram_data)), sharex=False, sharey=False)

cmap = plt.get_cmap('tab10')
c_pos, c_neg = cmap(0), cmap(1)

if len(histogram_data) == 1:
    axes = np.expand_dims(axes, axis=0)

for idx, (alpha, data) in enumerate(histogram_data.items()):
    ax_unf = axes[idx, 0]
    ax_fair = axes[idx, 1]
    
    y_u, y_f = data['y_u'], data['y_f']
    mask_pos, mask_neg = data['mask_pos'], data['mask_neg']

    bins = 50

    
    # Plot Unfair Histograms
    ax_unf.hist(y_u[mask_pos], bins=bins, density=True, alpha=0.5, color=c_pos, label=r'S = 1')
    ax_unf.hist(y_u[mask_neg], bins=bins, density=True, alpha=0.5, color=c_neg, label=r'S = 2')
    ax_unf.set_title(r"Unfair Predictions ($\alpha_0 = %.1f$)" % alpha)
    ax_unf.set_ylabel("Density")
    if idx == 0 :
        
        ax_unf.legend(loc='upper right')
    ax_unf.grid(axis='y', alpha=0.3)
    
  
    # Plot Fair Histograms (Barycenter)
    ax_fair.hist(y_f[mask_pos], bins=bins, density=True, alpha=0.5, color=c_pos, label=r'Fair | S = 1')
    ax_fair.hist(y_f[mask_neg], bins=bins, density=True, alpha=0.5, color=c_neg, label=r'Fair | S = 2')
    
    # Adding an outline for the overall barycenter distribution
    ax_fair.hist(y_f, bins=bins, density=True, histtype='step', linewidth=2, color='black', linestyle='--', label='Overall Barycenter')
    
    ax_fair.set_title(r"Fair Predictions ($\alpha_0 = %.1f$) | $W_1 = %.4f$ | KS = %.4f" % (alpha, data['w1_f'], data['ks_f']))
    if idx == 0 :
        ax_fair.legend(loc='upper right')
    ax_fair.grid(axis='y', alpha=0.3)

plt.suptitle("Distributions Before and After Fairness Correction (linear base regressor)", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


# %%
