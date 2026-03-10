# Test different unaware/aware regressors 
# and test differents base regresseurs/ mapping estimation
# visualisation step by step

# the "test Taturyan" part takes some time to run
#%%

import numpy as np
import ot 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, LogisticRegression
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from OTUnawareFairRegressor import OTUnawareFairRegressor 
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from sklearn.kernel_ridge import KernelRidge
from sklearn.ensemble import RandomForestRegressor
import sys
import os
from OTAwareFairRegressor import OTAwareFairRegressor
# %%
# Generate Data (X depends on S)
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

# visualisation of generated dataset 
n = 1000 
alpha_0 = 2
alpha_1 = 1 
x_scale = 1 
noise_scale = 0.3 
p = 0.5
X, Y, S = generate_linear_data(n = n, alpha_0 = alpha_0, alpha_1 = alpha_1, x_scale = x_scale, noise_scale = noise_scale, p = p)
X_train, X_test, Y_train, Y_test, S_train, S_test = train_test_split(X, Y, S, train_size = 0.6)


# %% 
# linearRegression
# regression for the whole dataset/majority group/minority group

std_reg = LinearRegression().fit(X_train, Y_train)
y_std = std_reg.predict(X_test)

X_train_maj = X_train[S_train == 1]
Y_train_maj = Y_train[S_train == 1]
X_test_maj = X_test[S_test == 1]
Y_test_maj = Y_test[S_test == 1]
std_reg_maj = LinearRegression().fit(X_train_maj, Y_train_maj)
y_std_maj = std_reg_maj.predict(X_test_maj)

X_train_min = X_train[S_train == 2]
Y_train_min = Y_train[S_train == 2]
X_test_min = X_test[S_test == 2]
Y_test_min = Y_test[S_test == 2]
std_reg_min = LinearRegression().fit(X_train_min, Y_train_min)
y_std_min = std_reg_min.predict(X_test_min)


# visualisation
cmap = plt.get_cmap('tab10')
color_maj = cmap(0)  # Color for S=1 (Orange)
color_min = cmap(1)  # Color for S=2 (Green)
color_all = 'black'  # Color for the unfair regressor

plt.figure(figsize=(10, 6))

plt.scatter(X[S == 1], Y[S == 1], color=color_maj, alpha=0.5, s=30, 
            label='Data S=1 (Majority)')

plt.scatter(X[S == 2], Y[S == 2], color=color_min, alpha=0.5, s=30, 
            label='Data S=2 (Minority)')

# Plot Regression Lines 
x_range_min = X.min() - 0.2
x_range_max = X.max() + 0.2
X_plot = np.linspace(x_range_min, x_range_max, 1000).reshape(-1, 1)

# Line for S=1
plt.plot(X_plot, std_reg_maj.predict(X_plot), color=color_maj, 
         linewidth=3, label='Regressor S=1')

# Line for S=2
plt.plot(X_plot, std_reg_min.predict(X_plot), color=color_min, 
         linewidth=3, label='Regressor S=2')

# Line for Unfair (Combined)
plt.plot(X_plot, std_reg.predict(X_plot), color=color_all, linestyle='--', 
         linewidth=2, label='Unfair Regressor (Combined)')

plt.title(" Bias in Generated Data (linear regressor)", fontsize=14)
plt.xlabel("Feature X")
plt.ylabel("Target Y")

plt.legend(frameon=True, loc='best')

plt.tight_layout()
plt.show()

# %%

# Fair regresseur
ot_reg = OTUnawareFairRegressor()
ot_reg.fit(X_train, Y_train, S_train)
y_fair = ot_reg.predict(X_test, prediction= "knn")


def plot_fairness_correction(X, y_unfair, y_fair, s_attr, save_path=None):
    """
    Visualizes the correction from unfair to fair predictions with connecting lines.
    
    Parameters:
    -----------
    X : array-like
        The input feature (X_test).
    y_unfair : array-like
        Predictions from the unfair model (y_std).
    y_fair : array-like
        Predictions from the fair model (y_fair).
    s_attr : array-like
        Sensitive attribute (S_test).
    save_path : str, optional
        Path to save the figure (e.g., 'results/plot.png').
    """
    
    x_flat = np.array(X).flatten()
    y_std_flat = np.array(y_unfair).flatten()
    y_fair_flat = np.array(y_fair).flatten()
    s_flat = np.array(s_attr).flatten()


    cmap = plt.get_cmap('tab10')
    c1 = cmap(0) # Blue
    c2 = cmap(1) # Orange

    # Robustly map the two groups to colors
    # We sort unique values so lower S (e.g., 1) gets Blue, higher S (e.g., 2) gets Orange
    unique_s = np.unique(s_flat)
    if len(unique_s) < 2:
        # Fallback if only 1 group exists
        s_val1, s_val2 = unique_s[0], unique_s[0]
    else:
        s_val1, s_val2 = unique_s[0], unique_s[1]
        

    colors = [c1 if s == s_val1 else c2 for s in s_flat]


    plt.figure(figsize=(10, 6))
    segments = np.column_stack((x_flat, y_std_flat, x_flat, y_fair_flat)).reshape(-1, 2, 2)
    lc = LineCollection(segments, colors='gray', alpha=0.3, linewidths=0.5, zorder=0)
    plt.gca().add_collection(lc)

    # Unfair Prediction (Circles)
    plt.scatter(x_flat, y_std_flat, c=colors, alpha=0.6, s=30, 
                marker='o', edgecolors='white', linewidth=0.5, zorder=1)

    # Fair Prediction (Stars)
    plt.scatter(x_flat, y_fair_flat, c=colors, alpha=0.9, s=80, 
                marker='*', edgecolors='white', linewidth=0.5, zorder=2)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=f'Group S={s_val1}',
               markerfacecolor=c1, markersize=10),
        Line2D([0], [0], marker='o', color='w', label=f'Group S={s_val2}',
               markerfacecolor=c2, markersize=10),
        
        Line2D([0], [0], color='white', label=' '),
        
        Line2D([0], [0], marker='o', color='w', label='Unfair Prediction',
               markerfacecolor='gray', markersize=8, alpha=0.7),
        Line2D([0], [0], marker='*', color='w', label='Fair Prediction',
               markerfacecolor='gray', markersize=12, alpha=0.9),
        
        # Correction Line
        Line2D([0], [0], color='gray', lw=1, label='Correction (Shift)'),
    ]

    plt.legend(handles=legend_elements, loc='best', frameon=True)

    plt.title(f"Fairness Correction: Group S={s_val1} vs S={s_val2}", fontsize=14)
    plt.xlabel("Feature (X)")
    plt.ylabel("Predicted Target (Y)")

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()


plot_fairness_correction(
    X=X_test, 
    y_unfair=y_std, 
    y_fair=y_fair, 
    s_attr=S_test
)


# %%

# Calculate KS Distance
def plot_ks_comparison(y_unfair, y_fair, s_attr, group_names=None, save_path=None):
    """
    Calculates KS statistics and plots distribution histograms for unfair vs fair predictions.
    
    Parameters:
    -----------
    y_unfair : array-like
        Predictions from the standard (unfair) model.
    y_fair : array-like
        Predictions from the fair (corrected) model.
    s_attr : array-like
        Sensitive attribute values (must contain exactly 2 unique groups).
    group_names : list of str, optional
        Custom names for the groups in the legend (e.g., ['Men', 'Women']).
        If None, defaults to 'Group {val}'.
    save_path : str, optional
        If provided, saves the figure to this path (e.g., './results/plot.png').
    """
    
    # 1. Setup Data & Groups
    y_u = np.array(y_unfair).flatten()
    y_f = np.array(y_fair).flatten()
    s = np.array(s_attr).flatten()
    
    # Automatically detect the two groups (e.g., 0/1 or 1/2)
    groups = np.unique(s)
    if len(groups) != 2:
        raise ValueError(f"Expected exactly 2 groups in s_attr, found {len(groups)}: {groups}")
    
    g1, g2 = groups[0], groups[1]
    
    # Default group names if not provided
    if group_names is None:
        labels = [f'Group {g1}', f'Group {g2}']
    else:
        labels = group_names

    # Unfair
    ks_std = ks_2samp(y_u[s == g1], y_u[s == g2])
    # Fair
    ks_fair = ks_2samp(y_f[s == g1], y_f[s == g2])

    print(f"KS Distance (Unfair): {ks_std.statistic:.4f} (p={ks_std.pvalue:.4e})")
    print(f"KS Distance (Fair):   {ks_fair.statistic:.4f} (p={ks_fair.pvalue:.4e})")

    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    c1, c2 = 'tab:blue', 'tab:orange'
    bins = 20
    alpha = 0.6

    # Unfair Distributions 
    axes[0].hist(y_u[s == g1], bins=bins, alpha=alpha, density=True, color=c1, label=labels[0])
    axes[0].hist(y_u[s == g2], bins=bins, alpha=alpha, density=True, color=c2, label=labels[1])
    
    axes[0].set_title(f"Unfair Regressor\nKS Distance: {ks_std.statistic:.3f}", fontsize=14)
    axes[0].set_xlabel("Predicted Y", fontsize=12)
    axes[0].set_ylabel("Density", fontsize=12)
    axes[0].legend()
    axes[0].grid(axis='y', linestyle=':', alpha=0.5)

    # Fair Distributions
    axes[1].hist(y_f[s == g1], bins=bins, alpha=alpha, density=True, color=c1, label=labels[0])
    axes[1].hist(y_f[s == g2], bins=bins, alpha=alpha, density=True, color=c2, label=labels[1])
    
    axes[1].set_title(f"Fair Regressor\nKS Distance: {ks_fair.statistic:.3f}", fontsize=14)
    axes[1].set_xlabel("Predicted Y", fontsize=12)
    axes[1].legend()
    axes[1].grid(axis='y', linestyle=':', alpha=0.5)

    plt.suptitle("Impact of Fairness Correction on Prediction Distributions", fontsize=16, y=1.02)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        
    plt.show()

plot_ks_comparison(
    y_unfair=y_std, 
    y_fair=y_fair, 
    s_attr=S_test, 
    group_names=['Majority (S=1)', 'Minority (S=2)']
)

# %%

def plot_fairness_shift(y_unfair, y_fair, s_attr, delta, n_samples=None, seed=42):
    """
    Visualizes the shift from unfair to fair predictions using a transport map style.
    
    Parameters:
    -----------
    y_unfair : array-like
        The original (unfair) predicted values.
    y_fair : array-like
        The corrected (fair) predicted values.
    s_attr : array-like
        The sensitive attribute.
    delta : array-like
        The 'cost' or magnitude of correction (y-axis in the plot).
    n_samples : int, optional
        Number of points to visualize. If None, plots all points.
    seed : int
        Random seed for sampling consistency.
    """
    
    # Standardize Inputs
    y_u = np.array(y_unfair).flatten()
    y_f = np.array(y_fair).flatten()
    s = np.array(s_attr).flatten()
    d = np.array(delta).flatten()
    

    if n_samples is not None and n_samples < len(y_u):
        np.random.seed(seed)
        indices = np.random.choice(len(y_u), n_samples, replace=False)
        y_u, y_f, s, d = y_u[indices], y_f[indices], s[indices], d[indices]


    cmap = plt.get_cmap('tab10')
    c_blue = cmap(0)  
    c_orange = cmap(1)
    
    # Map groups to colors automatically
    unique_groups = np.unique(s)
    group_colors = {unique_groups[0]: c_blue, unique_groups[1]: c_orange}
    
    # Create color list for the points
    point_colors = [group_colors[val] for val in s]

    # Plotting
    plt.figure(figsize=(10, 6))
    

    # Start: (Unfair Prediction, Delta)
    # End:   (Fair Prediction, 0)
    start_points = np.column_stack((y_u, d))
    end_points = np.column_stack((y_f, np.zeros_like(d)))
    
    segments = np.stack((start_points, end_points), axis=1)
    lc = LineCollection(segments, colors='gray', alpha=0.2, linewidths=0.8, zorder=0)
    plt.gca().add_collection(lc)
    
    # Unfair (Start) - Stars
    plt.scatter(y_u, d, c=point_colors, s=50, marker='*', 
                alpha=0.7, edgecolors='white', linewidth=0.5, zorder=1)
    
    # Fair (End) - Circles (Projected onto y=0)
    plt.scatter(y_f, np.zeros_like(d), c=point_colors, s=50, marker='o', 
                alpha=0.9, edgecolors='white', linewidth=0.5, zorder=2)

    legend_elements = [
        # Group Headers
        Line2D([0], [0], marker='o', color='w', label=f'Group {unique_groups[0]}',
               markerfacecolor=c_blue, markersize=10),
        Line2D([0], [0], marker='o', color='w', label=f'Group {unique_groups[1]}',
               markerfacecolor=c_orange, markersize=10),
        Line2D([0], [0], color='white', label=' '), # Spacer
        
        # Shape Meanings
        Line2D([0], [0], marker='*', color='w', label='Unfair Prediction',
               markerfacecolor='gray', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Fair Prediction',
               markerfacecolor='gray', markersize=10),
    ]

    plt.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.3)
    plt.xlabel("Predicted Value ($y$)")
    plt.ylabel("delta ($\Delta(x)$)")
    plt.title(f"Fairness Correction Map (Sampled {len(y_u)} points)")
    plt.legend(handles=legend_elements, loc='upper right', frameon=True)
    
    plt.tight_layout()
    plt.savefig("./results/generic_data_unaware_correction_line.png") 
    plt.show()


plot_fairness_shift(
    y_unfair = y_std, 
    y_fair = y_fair, 
    s_attr = S_test, 
    delta = ot_reg.delta_predict, 
    n_samples = 50 
)
# %%


# Gaussian process regression 
# regression for the whole dataset/majority group/minority group

kernel = 2 * RBF(length_scale=3.0, length_scale_bounds=(1e-2, 1e2))
gp_reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=noise_scale**2).fit(X_train, Y_train)
y_gp = gp_reg.predict(X_test)

X_train_maj = X_train[S_train == 1]
Y_train_maj = Y_train[S_train == 1]
X_test_maj = X_test[S_test == 1]
Y_test_maj = Y_test[S_test == 1]

gp_reg_maj = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=noise_scale**2).fit(X_train_maj, Y_train_maj)
X_test_maj_sorted = np.sort(X_test_maj, axis = 0)
y_gp_maj_sorted = gp_reg_maj.predict(X_test_maj_sorted)

X_train_min = X_train[S_train == 2]
Y_train_min = Y_train[S_train == 2]
X_test_min = X_test[S_test == 2]
Y_test_min = Y_test[S_test == 2]
gp_reg_min = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=noise_scale**2).fit(X_train_min, Y_train_min)
X_test_min_sorted = np.sort(X_test_min, axis =0 )
y_gp_min_sorted = gp_reg_min.predict(X_test_min_sorted)


# visualisation
cmap = plt.get_cmap('tab10')
color_maj = cmap(0)  # Color for S=1 (Orange)
color_min = cmap(1)  # Color for S=2 (Green)
color_all = 'black'  # Color for the unfair regressor

plt.figure(figsize=(10, 6))

# Plot Data Points (Split by group for the legend)
plt.scatter(X[S == 1], Y[S == 1], color=color_maj, alpha=0.5, s=30, 
            label='Data S=1 (Majority)')

plt.scatter(X[S == 2], Y[S == 2], color=color_min, alpha=0.5, s=30, 
            label='Data S=2 (Minority)')

# Plot Regression Lines 
# Create X range for smooth lines
x_range_min = X.min() - 0.2
x_range_max = X.max() + 0.2
X_plot = np.linspace(x_range_min, x_range_max, 1000).reshape(-1, 1)

# Line for S=1

plt.plot(X_test_maj_sorted, y_gp_maj_sorted, color=color_maj, 
         linewidth=3, label='Regressor S=1')

# Line for S=2
plt.plot(X_test_min_sorted, y_gp_min_sorted, color=color_min, 
         linewidth=3, label='Regressor S=2')

# Line for Unfair (Combined)
plt.plot(X_plot, gp_reg.predict(X_plot), color=color_all, linestyle='--', 
         linewidth=2, label='Unfair Regressor (Combined)')

plt.title(" Bias in Generated Data (Gaussian process regressor)", fontsize=14)
plt.xlabel("Feature X")
plt.ylabel("Target Y")

# Legend
plt.legend(frameon=True, loc='best')

plt.tight_layout()
plt.show()

# %%

# Fair regresseur with random forest mapping estimation
kernel_krr = KernelRidge(kernel='rbf', alpha=0.1, gamma=0.3)
regr_rf = RandomForestRegressor(max_depth=4, random_state=0)
ot_reg_gp = OTUnawareFairRegressor(base_regressor= gp_reg,kernel_krr=kernel_krr, n_neighbors= 5, random_forest=regr_rf)
ot_reg_gp.fit(X_train, Y_train, S_train)

# krr regressor
y_gp_fair_krr = ot_reg_gp.predict(X_test, prediction= "krr")

# knn regressor
y_gp_fair_knn = ot_reg_gp.predict(X_test, prediction= "knn")

# random forest regressor
y_gp_fair_rf = ot_reg_gp.predict(X_test, prediction= "random_forest")


def plot_fairness_correction(X, y_unfair, y_fair, s_attr, save_path=None, method_name="knn"):
    """
    Visualizes the correction from unfair to fair predictions with connecting lines.
    
    Parameters:
    -----------
    X : array-like
        The input feature (X_test).
    y_unfair : array-like
        Predictions from the unfair model (y_std).
    y_fair : array-like
        Predictions from the fair model (y_fair).
    s_attr : array-like
        Sensitive attribute (S_test).
    save_path : str, optional
        Path to save the figure (e.g., 'results/plot.png').
    """
    
    x_flat = np.array(X).flatten()
    y_std_flat = np.array(y_unfair).flatten()
    y_fair_flat = np.array(y_fair).flatten()
    s_flat = np.array(s_attr).flatten()

    cmap = plt.get_cmap('tab10')
    c1 = cmap(0) # Blue
    c2 = cmap(1) # Orange

    unique_s = np.unique(s_flat)
    if len(unique_s) < 2:
        # Fallback if only 1 group exists
        s_val1, s_val2 = unique_s[0], unique_s[0]
    else:
        s_val1, s_val2 = unique_s[0], unique_s[1]
        

    colors = [c1 if s == s_val1 else c2 for s in s_flat]


    plt.figure(figsize=(6, 4))
    segments = np.column_stack((x_flat, y_std_flat, x_flat, y_fair_flat)).reshape(-1, 2, 2)
    lc = LineCollection(segments, colors='gray', alpha=0.3, linewidths=0.5, zorder=0)
    plt.gca().add_collection(lc)

    # Unfair Prediction (Circles)
    plt.scatter(x_flat, y_std_flat, c=colors, alpha=0.6, s=30, 
                marker='o', edgecolors='white', linewidth=0.5, zorder=1)

    # Fair Prediction (Stars)
    plt.scatter(x_flat, y_fair_flat, c=colors, alpha=0.9, s=80, 
                marker='*', edgecolors='white', linewidth=0.5, zorder=2)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=f'Group S={s_val1}',
               markerfacecolor=c1, markersize=10),
        Line2D([0], [0], marker='o', color='w', label=f'Group S={s_val2}',
               markerfacecolor=c2, markersize=10),
        
        Line2D([0], [0], color='white', label=' '),
        
        Line2D([0], [0], marker='o', color='w', label='Unfair Prediction',
               markerfacecolor='gray', markersize=8, alpha=0.7),
        Line2D([0], [0], marker='*', color='w', label='Fair Prediction',
               markerfacecolor='gray', markersize=12, alpha=0.9),
        
        Line2D([0], [0], color='gray', lw=1, label='Correction (Shift)'),
    ]

    plt.legend(handles=legend_elements, loc='best', frameon=True)

    plt.title(f"Fairness Correction ({method_name})", fontsize=14)
    plt.xlabel("Feature (X)")
    plt.ylabel("Predicted Target (Y)")

    # Clean look
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure saved to {save_path}")

    plt.show()

# plot for different mapping estimation
plot_fairness_correction(
    X=X_test, 
    y_unfair=y_gp, 
    y_fair=y_gp_fair_knn, 
    s_attr=S_test, 
    save_path="./results/fairness_correction_scatter_gp.png"
)
plot_fairness_correction(
    X=X_test, 
    y_unfair=y_gp, 
    y_fair=y_gp_fair_krr, 
    s_attr=S_test, 
    method_name="krr",
    save_path="./results/fairness_correction_scatter_gp.png"
)
plot_fairness_correction(
    X=X_test, 
    y_unfair=y_gp, 
    y_fair=y_gp_fair_rf, 
    s_attr=S_test, 
    method_name="random forest",
    save_path="./results/fairness_correction_scatter_gp.png"
)

# visualize aware correction

aware_model = OTAwareFairRegressor(gp_reg).fit(X_train, Y_train, S_train)
y_fair_aware = aware_model.predict(X_test, S_test)
y_fair_aware_derived = aware_model.predict(X_test)
plot_fairness_correction(
    X=X_test, 
    y_unfair=y_gp, 
    y_fair=y_fair_aware, 
    s_attr=S_test, 
    method_name="aware",
    save_path="./results/fairness_correction_scatter_gp.png"
)
plot_fairness_correction(
    X=X_test, 
    y_unfair=y_gp, 
    y_fair=y_fair_aware_derived, 
    s_attr=S_test, 
    method_name="aware derived",
    save_path="./results/fairness_correction_scatter_gp.png"
)

# %%
# test Taturyan
# ATTENTION: this may take some time

current_dir = os.getcwd()
#print(f"Notebook is running in: {current_dir}")

folder_path = os.path.abspath(os.path.join(current_dir, 'unaware-fair-reg-3rd-method'))
#print(f"Looking for module folder at: {folder_path}")
#print(f"Does this folder exist? {os.path.exists(folder_path)}")

if folder_path not in sys.path:
    sys.path.insert(0, folder_path)
from FairReg import FairReg

# Pre-fit the Base Regressor and the Proxy Classifier
# FairReg requires these to be already fitted on the training data
proxy_classifier = LogisticRegression()
proxy_classifier.fit(X_train, S_train)

# Extract required parameters for FairReg
# B: Bound on the target variable (max absolute value of y)
B_val = np.max(np.abs(Y_train)) 

# K: Number of sensitive attribute groups
unique_groups = np.unique(S_train)
K_val = len(unique_groups)

# p: Frequencies of each sensitive group in the training data
p_val = [np.mean(S_train == s) for s in unique_groups]

# eps: Epsilon thresholds for demographic parity (tolerance for unfairness)
eps_val = [0.00001 for _ in range(K_val)] 

# T: Number of iterations for the stochastic gradient descent
T_val = 1000000

# Initialize the FairReg model
fair_reg_taturyan = FairReg(
    base_method=gp_reg,
    classifier=proxy_classifier,
    B=B_val,
    K=K_val,
    p=p_val,
    eps=eps_val,
    T=T_val
)

# Fit the fairness weights (w_est) using X_train
fair_reg_taturyan.fit(X_train)

# Predict on the test set
y_pred_taturyan = fair_reg_taturyan.predict(X_test)

# plot fairness correction plan
plot_fairness_correction(
    X=X_test, 
    y_unfair=y_gp, 
    y_fair=y_pred_taturyan, 
    s_attr=S_test, 
    method_name="unaware taturyan",
    save_path="./results/fairness_correction_scatter_gp.png"
)

# %% 
# plot histogram

def plot_ks_hist_with_name(y_unfair, y_fair, s_attr, group_names=None, save_path=None,regressor_name = 'knn'):
    """
    Calculates w1 et KS statistics and plots distribution histograms for fair predictions.
    
    Parameters:
    -----------
   y_fair : array-like
        Predictions from the fair (corrected) model.
    s_attr : array-like
        Sensitive attribute values (must contain exactly 2 unique groups).
    group_names : list of str, optional
        Custom names for the groups in the legend (e.g., ['Men', 'Women']).
        If None, defaults to 'Group {val}'.
    save_path : str, optional
        If provided, saves the figure to this path (e.g., './results/plot.png').
    """
    

    y_f = np.array(y_fair).flatten()
    s = np.array(s_attr).flatten()
    
    # Automatically detect the two groups (e.g., 0/1 or 1/2)
    groups = np.unique(s)
    if len(groups) != 2:
        raise ValueError(f"Expected exactly 2 groups in s_attr, found {len(groups)}: {groups}")
    
    g1, g2 = groups[0], groups[1]
    
    # Default group names if not provided
    if group_names is None:
        labels = [f'Group {g1}', f'Group {g2}']
    else:
        labels = group_names

    
    # Fair
    ks_fair = ks_2samp(y_f[s == g1], y_f[s == g2])
    w1 = ot.lp. wasserstein_1d( y_f[s == g1], y_f[s == g2], np.ones_like(y_f[s == g1])/len(y_f[s == g1]),np.ones_like(y_f[s == g2])/len(y_f[s == g2]))
    print(f"KS Distance (Fair):   {ks_fair.statistic:.4f} (p={ks_fair.pvalue:.4e})")

    fig, axes = plt.subplots(1, 1, figsize=(6, 4))

    c1, c2 = 'tab:blue', 'tab:orange'
    bins = 20
    alpha = 0.6


    axes.hist(y_f[s == g1], bins=bins, alpha=alpha, density=True, color=c1, label=labels[0])
    axes.hist(y_f[s == g2], bins=bins, alpha=alpha, density=True, color=c2, label=labels[1])
    
    axes.set_title(f"fair regressor ({regressor_name})\n w1 : {w1:.3f}, KS Distance: {ks_fair.statistic:.3f}", fontsize=14)
    axes.set_xlabel("Predicted Y", fontsize=12)
    axes.set_ylabel("Density", fontsize=12)
    axes.legend()
    axes.grid(axis='y', linestyle=':', alpha=0.5)


    if save_path:

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        
    plt.show()

plot_ks_hist_with_name(
    y_unfair=y_gp, 
    y_fair=y_gp_fair_rf, 
    s_attr=S_test, 
    group_names=['Majority (S=1)', 'Minority (S=2)'], # Optional custom labels
    save_path="./results/generic_data_unaware_KS_gp_rf.png",
    regressor_name= "random forest"
)
plot_ks_hist_with_name(
    y_unfair=y_gp, 
    y_fair=y_gp_fair_krr, 
    s_attr=S_test, 
    group_names=['Majority (S=1)', 'Minority (S=2)'], # Optional custom labels
    save_path="./results/generic_data_unaware_KS_gp_rf.png",
    regressor_name= "krr"
)
plot_ks_hist_with_name(
    y_unfair=y_gp, 
    y_fair=y_gp_fair_knn, 
    s_attr=S_test, 
    group_names=['Majority (S=1)', 'Minority (S=2)'], # Optional custom labels
    save_path="./results/generic_data_unaware_KS_gp_rf.png",
    regressor_name= "knn"
)

# %% 
# plot unfair histogram 

def plot_ks_hist_unfair(y_unfair, s_attr, group_names=None, save_path=None):
    """
    Calculates w1 et KS statistics and plots distribution histograms for fair predictions.
    
    Parameters:
    -----------
   y_unfair : array-like
        Predictions from the unfair model.
    s_attr : array-like
        Sensitive attribute values (must contain exactly 2 unique groups).
    group_names : list of str, optional
        Custom names for the groups in the legend (e.g., ['Men', 'Women']).
        If None, defaults to 'Group {val}'.
    save_path : str, optional
        If provided, saves the figure to this path (e.g., './results/plot.png').
    """
    

    y_u = np.array(y_unfair).flatten()
    s = np.array(s_attr).flatten()
    
    # Automatically detect the two groups (e.g., 0/1 or 1/2)
    groups = np.unique(s)
    if len(groups) != 2:
        raise ValueError(f"Expected exactly 2 groups in s_attr, found {len(groups)}: {groups}")
    
    g1, g2 = groups[0], groups[1]
    
    # Default group names if not provided
    if group_names is None:
        labels = [f'Group {g1}', f'Group {g2}']
    else:
        labels = group_names

    # Calculate KS Statistics
    # unfair
    ks_unfair = ks_2samp(y_u[s == g1], y_u[s == g2])
    w1 = ot.lp. wasserstein_1d( y_u[s == g1], y_u[s == g2], np.ones_like(y_u[s == g1])/len(y_u[s == g1]),np.ones_like(y_u[s == g2])/len(y_u[s == g2]))
    print(f"KS Distance (Unfair):   {ks_unfair.statistic:.4f}")

    # Visualization
    fig, axes = plt.subplots(1, 1, figsize=(6, 4))

    c1, c2 = 'tab:blue', 'tab:orange'
    bins = 20
    alpha = 0.6


    axes.hist(y_u[s == g1], bins=bins, alpha=alpha, density=True, color=c1, label=labels[0])
    axes.hist(y_u[s == g2], bins=bins, alpha=alpha, density=True, color=c2, label=labels[1])
    
    axes.set_title(f"unfair regressor \n W1 : {w1:.3f}, KS Distance: {ks_unfair.statistic:.3f}", fontsize=14)
    axes.set_xlabel("Predicted Y", fontsize=12)
    axes.set_ylabel("Density", fontsize=12)
    axes.legend()
    axes.grid(axis='y', linestyle=':', alpha=0.5)


    if save_path:

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
        
    plt.show()

plot_ks_hist_unfair(
    y_unfair=y_gp, 

    s_attr=S_test, 
    group_names=['Majority (S=1)', 'Minority (S=2)'], # Optional custom labels
    save_path="./results/generic_data_unaware_unfair.png"
)

# %%
# comparison for different mapping estimations
plot_ks_comparison(
    y_unfair=y_gp, 
    y_fair=y_gp_fair_knn, 
    s_attr=S_test, 
    group_names=['Majority (S=1)', 'Minority (S=2)'], # Optional custom labels
    save_path="./results/generic_data_unaware_KS_gp_knn.png"
)
plot_ks_comparison(
    y_unfair=y_gp, 
    y_fair=y_gp_fair_krr, 
    s_attr=S_test, 
    group_names=['Majority (S=1)', 'Minority (S=2)'], # Optional custom labels
    save_path="./results/generic_data_unaware_KS_gp_krr.png"
)
plot_ks_comparison(
    y_unfair=y_gp, 
    y_fair=y_gp_fair_rf, 
    s_attr=S_test, 
    group_names=['Majority (S=1)', 'Minority (S=2)'], # Optional custom labels
    save_path="./results/generic_data_unaware_KS_gp_rf.png"
)

plot_fairness_shift(
    y_unfair = y_gp, 
    y_fair = y_gp_fair_krr, 
    s_attr = S_test, 
    delta = ot_reg_gp.delta_predict, 
    n_samples = 50 
)

plot_fairness_shift(
    y_unfair = y_gp, 
    y_fair = y_gp_fair_knn, 
    s_attr = S_test, 
    delta = ot_reg_gp.delta_predict, 
    n_samples = 50 
)
plot_fairness_shift(
    y_unfair = y_gp, 
    y_fair = y_gp_fair_rf, 
    s_attr = S_test, 
    delta = ot_reg_gp.delta_predict, 
    n_samples = 50 
)
# %%