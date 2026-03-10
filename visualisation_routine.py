# Visualize the data of each step during 
# the construction of OT fair unaware regressor 
# as well as a visualisation with histogram with different alpha_0
# This file serves like a visualisation template

#%%
import numpy as np
import ot 
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from OTUnawareFairRegressor import OTUnawareFairRegressor 
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF
from OTAwareFairRegressor import OTAwareFairRegressor
from sklearn.metrics import mean_squared_error
from scipy.stats import wasserstein_distance
# %%
# Generate Data (X depends on S)
def generate_linear_data(n , alpha_0=2, alpha_1=1, p = 0.5, x_scale = 1, noise_scale = 1, seed = 42):
    """
    Generate 1D linear data.  
    (X = normal(0, x_scale) + alpha_0 * S, Y = X + alpha_1 * S + normal(0, noise_scale, n) )
    S = 1 for majority
    S = 2 for minority

    :param n: size of dataset
    :param alpha_0: X = normal(0, x_scale) + alpha_0*S 
    :param alpha_1: Y = X + alpha_1*S 
    :param p: probability for S = 2 (parameter in random.binomial)
    :param x_scale: X = normal(0, x_scale) + alpha_0*S 
    :param seed: random seed
    """
    np.random.seed(seed)
    S = np.random.binomial(1, p, n)+1 # Binary sensitive attributes

    X = np.random.normal(0, x_scale, n) - alpha_0 * S
    X = X.reshape(-1, 1)

    # Y depends on X and S 
    Y = -alpha_1*S + 0.5 * X.flatten() + np.random.normal(0, noise_scale, n)
   
    return X, Y, S


# %%
# visualisation of generated dataset 
n = 1000 
alpha_0 = 2
alpha_1 = 1 
x_scale = 1 
noise_scale = 0.3 
X, Y, S = generate_linear_data(n = n, alpha_0 = alpha_0, alpha_1 = alpha_1, x_scale = x_scale, noise_scale = noise_scale)

X_train, X_test, Y_train, Y_test, S_train, S_test = train_test_split(X, Y, S, train_size = 0.7)
X_train_maj = X_train[S_train == 1]
Y_train_maj = Y_train[S_train == 1]
X_test_maj = X_test[S_test == 1]
Y_test_maj = Y_test[S_test == 1]

X_train_min = X_train[S_train == 2]
Y_train_min = Y_train[S_train == 2]
X_test_min = X_test[S_test == 2]
Y_test_min = Y_test[S_test == 2]

# %%
# linear regression (unfair)
# for the whole dataset/majority group/minority group
std_reg = LinearRegression().fit(X_train, Y_train)
y_std = std_reg.predict(X_test)


std_reg_maj = LinearRegression().fit(X_train_maj, Y_train_maj)
y_std_maj = std_reg_maj.predict(X_test_maj)


std_reg_min = LinearRegression().fit(X_train_min, Y_train_min)
y_std_min = std_reg_min.predict(X_test_min)

cmap = plt.get_cmap('tab10')
color_maj = cmap(0)  # Color for S=1 (Orange)
color_min = cmap(1)  # Color for S=2 (Green)
color_all = 'black'  # Color for the unfair regressor

plt.figure(figsize=(6, 4))
plt.scatter(X[S == 1], Y[S == 1], color=color_maj, alpha=0.5, s=30, 
            label='S=1')

plt.scatter(X[S == 2], Y[S == 2], color=color_min, alpha=0.5, s=30, 
            label='S=2')
plt.xlabel("Feature X")
plt.ylabel("Target Y")
plt.legend(frameon=True, loc='best')
plt.title("Generated toy example")
plt.show()


#%%
# data visualisation with linear regression


plt.figure(figsize=(5, 3.5))
plt.scatter(X[S == 1], Y[S == 1], color=color_maj, alpha=0.5, s=30, 
            label='S=1')
plt.scatter(X[S == 2], Y[S == 2], color=color_min, alpha=0.5, s=30, 
            label='S=2')

#regression lines 
x_range_min = X.min() - 0.2
x_range_max = X.max() + 0.2
X_plot = np.linspace(x_range_min, x_range_max, 1000).reshape(-1, 1)


plt.plot(X_plot, std_reg_maj.predict(X_plot), color=color_maj, 
         linewidth=3, label='Regressor S=1')

plt.plot(X_plot, std_reg_min.predict(X_plot), color=color_min, 
         linewidth=3, label='Regressor S=2')

plt.plot(X_plot, std_reg.predict(X_plot), color=color_all, linestyle='--', 
         linewidth=2, label='Unfair "bayes" regressor')

plt.title("linear regression", fontsize=14)
plt.xlabel("Feature X")
plt.ylabel("Target Y")

plt.legend(frameon=True, loc='best')
plt.show()



# %%
# Gaussian process regression (unfair)
# regression for the whole dataset/majority group/minority group

kernel = 2 * RBF(length_scale=3.0, length_scale_bounds=(1e-2, 1e2))
gp_reg = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=noise_scale**2).fit(X_train, Y_train)
y_gp = gp_reg.predict(X_test)

gp_reg_maj = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=noise_scale**2).fit(X_train_maj, Y_train_maj)
X_test_maj_sorted = np.sort(X_test_maj, axis = 0)
y_gp_maj_sorted = gp_reg_maj.predict(X_test_maj_sorted)

gp_reg_min = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9, alpha=noise_scale**2).fit(X_train_min, Y_train_min)
X_test_min_sorted = np.sort(X_test_min, axis =0 )
y_gp_min_sorted = gp_reg_min.predict(X_test_min_sorted)


# visualisation
plt.figure(figsize=(5, 4))
cmap = plt.get_cmap('tab10')
color_maj = cmap(0)  # Color for S=1 (Orange)
color_min = cmap(1)  # Color for S=2 (Green)
color_all = 'black'  # Color for the unfair regressor

#data points plot
plt.scatter(X[S == 1], Y[S == 1], color=color_maj, alpha=0.5, s=30, 
            label='S=1')

plt.scatter(X[S == 2], Y[S == 2], color=color_min, alpha=0.5, s=30, 
            label='S=2')


x_range_min = X.min() - 0.2
x_range_max = X.max() + 0.2
X_plot = np.linspace(x_range_min, x_range_max, 1000).reshape(-1, 1)

plt.plot(X_test_maj_sorted, y_gp_maj_sorted, color=color_maj, 
         linewidth=3, label='Regressor S=1')


plt.plot(X_test_min_sorted, y_gp_min_sorted, color=color_min, 
         linewidth=3, label='Regressor S=2')

plt.plot(X_plot, gp_reg.predict(X_plot), color=color_all, linestyle='--', 
         linewidth=2, label='Unfair "bayes" regressor')

plt.title("gaussian process regression")
plt.xlabel("Feature X")
plt.ylabel("Target Y")
plt.legend(frameon=True, loc='best')
plt.show()


# %%
# Fair unaware regresseur using linear regression for eta 
# and knn for mapping estimation
ot_reg = OTUnawareFairRegressor()
ot_reg.fit(X_train, Y_train, S_train)
y_fair = ot_reg.predict(X_test, prediction="knn")

# %%
# Fair aware regresseur using linear regression for eta
aware_model = OTAwareFairRegressor(LinearRegression()).fit(X_train, Y_train, S_train)
y_fair_aware = aware_model.predict(X_test, S_test)


# %% 
# visualisation of fairness correction

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

    unique_s = np.unique(s_flat)
    if len(unique_s) < 2:
        s_val1, s_val2 = unique_s[0], unique_s[0]
    else:
        s_val1, s_val2 = unique_s[0], unique_s[1]
        

    colors = [c1 if s == s_val1 else c2 for s in s_flat]


    plt.figure(figsize=(10, 6))
    segments = np.column_stack((x_flat, y_std_flat, x_flat, y_fair_flat)).reshape(-1, 2, 2)
    lc = LineCollection(segments, colors='gray', alpha=0.3, linewidths=0.5, zorder=0)
    plt.gca().add_collection(lc)

    #unfair predictions (circles)
    plt.scatter(x_flat, y_std_flat, c=colors, alpha=0.6, s=30, 
                marker='o', edgecolors='white', linewidth=0.5, zorder=1)

    #fair predictions (stars)
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

    plt.title(f"Fairness Correction: Group S={s_val1} vs S={s_val2}", fontsize=14)
    plt.xlabel("Feature (X)")
    plt.ylabel("Predicted Target (Y)")

    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    plt.grid(True, linestyle=':', alpha=0.5)
    plt.tight_layout()

    if save_path:
        #creating directory if it doesn't exist
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
# Calculate KS Distance and plot histograms
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
    
    y_u = np.array(y_unfair).flatten()
    y_f = np.array(y_fair).flatten()
    s = np.array(s_attr).flatten()
    
    # detect the two groups 
    groups = np.unique(s)
    if len(groups) != 2:
        raise ValueError(f"Expected exactly 2 groups in s_attr, found {len(groups)}: {groups}")
    
    g1, g2 = groups[0], groups[1]
    

    if group_names is None:
        labels = [f'Group {g1}', f'Group {g2}']
    else:
        labels = group_names

    #KS statistics
    #Unfair
    ks_std = ks_2samp(y_u[s == g1], y_u[s == g2])
    #Fair
    ks_fair = ks_2samp(y_f[s == g1], y_f[s == g2])

    print(f"KS Distance (Unfair): {ks_std.statistic:.4f} (p={ks_std.pvalue:.4e})")
    print(f"KS Distance (Fair):   {ks_fair.statistic:.4f} (p={ks_fair.pvalue:.4e})")

    #Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
    
    c1, c2 = 'tab:blue', 'tab:orange'
    bins = 20
    alpha = 0.6

    #Unfair Distributions 
    axes[0].hist(y_u[s == g1], bins=bins, alpha=alpha, density=True, color=c1, label=labels[0])
    axes[0].hist(y_u[s == g2], bins=bins, alpha=alpha, density=True, color=c2, label=labels[1])
    
    axes[0].set_title(f"Unfair Regressor\nKS Distance: {ks_std.statistic:.3f}", fontsize=14)
    axes[0].set_xlabel("Predicted Y", fontsize=12)
    axes[0].set_ylabel("Density", fontsize=12)
    axes[0].legend()
    axes[0].grid(axis='y', linestyle=':', alpha=0.5)

    #Fair Distributions
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
def plot_fairness_shift_with_hist(y_unfair, y_fair, s_attr, delta, n_samples=None, seed=42, save_path = None):
    """
    Visualizes the shift from unfair to fair predictions using a transport map style, alongside the initial conditional distributions and the final barycenter.
    """
    
    y_u_full = np.array(y_unfair).flatten()
    y_f_full = np.array(y_fair).flatten()
    s_full = np.array(s_attr).flatten()
    d_full = np.array(delta).flatten()
    
    unique_groups = np.unique(s_full)
    
    if n_samples is not None and n_samples < len(y_u_full):
        np.random.seed(seed)
        indices = np.random.choice(len(y_u_full), n_samples, replace=False)
        y_u, y_f, s, d = y_u_full[indices], y_f_full[indices], s_full[indices], d_full[indices]
    else:
        y_u, y_f, s, d = y_u_full, y_f_full, s_full, d_full

    cmap = plt.get_cmap('tab10')
    c_blue = cmap(0)  
    c_orange = cmap(1)
    group_colors = {unique_groups[0]: c_blue, unique_groups[1]: c_orange}
    point_colors = [group_colors[val] for val in s]

    fig = plt.figure(figsize=(10, 8))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 2.5], hspace=0.05)
    
    ax_hist = fig.add_subplot(gs[0])
    ax_scatter = fig.add_subplot(gs[1], sharex=ax_hist)

    group_0_mask = s_full == unique_groups[0]
    group_1_mask = s_full == unique_groups[1]
    
    # initial conditional distributions (unfair)
    ax_hist.hist(y_u_full[group_0_mask], bins=40, density=True, alpha=0.4, 
                 color=c_blue, label=f'Initial | S={unique_groups[0]}')
    ax_hist.hist(y_u_full[group_1_mask], bins=40, density=True, alpha=0.4, 
                 color=c_orange, label=f'Initial | S={unique_groups[1]}')
    
    # barycenter distribution (fair - target)
    ax_hist.hist(y_f_full, bins=40, density=True, histtype='step', 
                 linewidth=2, color='black', linestyle='--', label='Barycenter (Fair)')
    
    ax_hist.set_ylabel("Density")
    ax_hist.legend(loc='upper right')
    ax_hist.tick_params(labelbottom=False)
    ax_hist.grid(axis='x', alpha=0.3)
    #plt.show()


    #transport map

    start_points = np.column_stack((y_u, d))
    end_points = np.column_stack((y_f, np.zeros_like(d)))
    
    segments = np.stack((start_points, end_points), axis=1)
    lc = LineCollection(segments, colors='gray', alpha=0.3, linewidths=0.8, zorder=0)
    ax_scatter.add_collection(lc)
    
    #unfair stars
    ax_scatter.scatter(y_u, d, c=point_colors, s=60, marker='*', 
                       alpha=0.8, edgecolors='white', linewidth=0.5, zorder=1)
    
    #fair circles
    ax_scatter.scatter(y_f, np.zeros_like(d), c=point_colors, s=50, marker='o', 
                       alpha=0.9, edgecolors='white', linewidth=0.5, zorder=2)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=f'Group {unique_groups[0]}',
               markerfacecolor=c_blue, markersize=10),
        Line2D([0], [0], marker='o', color='w', label=f'Group {unique_groups[1]}',
               markerfacecolor=c_orange, markersize=10),
        Line2D([0], [0], color='white', label=' '), 
        Line2D([0], [0], marker='*', color='w', label='Unfair Prediction',
               markerfacecolor='gray', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Fair Prediction',
               markerfacecolor='gray', markersize=10),
    ]

    ax_scatter.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax_scatter.set_xlabel("Predicted Value ($y$)")
    ax_scatter.set_ylabel(r"$\Delta(x)$ (Correction Cost)")
    ax_scatter.legend(handles=legend_elements, loc='upper right', frameon=True)
    ax_scatter.grid(alpha=0.2)
    
    fig.suptitle(f"Fairness Correction Transport Map (Sampled {len(y_u)} points)", y=0.95)
    

    if save_path:
        plt.savefig("./results/generic_data_unaware_correction_line_with_hist.png", dpi=300) 
    plt.show()

plot_fairness_shift_with_hist(
    y_unfair = y_std, 
    y_fair = y_fair, 
    s_attr = S_test, 
    delta = ot_reg.delta_predict, 
    n_samples = 100
)


# %%
# visualize only the transport plan
def plot_fairness_plan(y_unfair, y_fair, s_attr, delta, n_samples=None, seed=42):
    """
    Visualizes the shift from unfair to fair predictions using a transport map style,
    alongside the initial conditional distributions and the final barycenter.
    """
    
    y_u_full = np.array(y_unfair).flatten()
    y_f_full = np.array(y_fair).flatten()
    s_full = np.array(s_attr).flatten()
    d_full = np.array(delta).flatten()
    
    unique_groups = np.unique(s_full)
    
    if n_samples is not None and n_samples < len(y_u_full):
        np.random.seed(seed)
        indices = np.random.choice(len(y_u_full), n_samples, replace=False)
        y_u, y_f, s, d = y_u_full[indices], y_f_full[indices], s_full[indices], d_full[indices]
    else:
        y_u, y_f, s, d = y_u_full, y_f_full, s_full, d_full

    cmap = plt.get_cmap('tab10')
    c_blue = cmap(0)  
    c_orange = cmap(1)
    group_colors = {unique_groups[0]: c_blue, unique_groups[1]: c_orange}
    point_colors = [group_colors[val] for val in s]

    fig = plt.figure(figsize=(8, 3))
    # Create two rows: top for histograms (height 1), bottom for scatter (height 2.5)
    gs = fig.add_gridspec(1, 1)
    
    ax_scatter = fig.add_subplot(gs[0])

   #transport map
    start_points = np.column_stack((y_u, d))
    end_points = np.column_stack((y_f, np.zeros_like(d)))
    
    segments = np.stack((start_points, end_points), axis=1)
    lc = LineCollection(segments, colors='gray', alpha=0.3, linewidths=0.8, zorder=0)
    ax_scatter.add_collection(lc)
    
    ax_scatter.scatter(y_u, d, c=point_colors, s=60, marker='*', 
                       alpha=0.8, edgecolors='white', linewidth=0.5, zorder=1)
    
    ax_scatter.scatter(y_f, np.zeros_like(d), c=point_colors, s=50, marker='o', 
                       alpha=0.9, edgecolors='white', linewidth=0.5, zorder=2)

    legend_elements = [
        Line2D([0], [0], marker='o', color='w', label=f'Group {unique_groups[0]}',
               markerfacecolor=c_blue, markersize=10),
        Line2D([0], [0], marker='o', color='w', label=f'Group {unique_groups[1]}',
               markerfacecolor=c_orange, markersize=10),
        Line2D([0], [0], color='white', label=' '), 
        Line2D([0], [0], marker='*', color='w', label='Unfair Prediction',
               markerfacecolor='gray', markersize=12),
        Line2D([0], [0], marker='o', color='w', label='Fair Prediction',
               markerfacecolor='gray', markersize=10),
    ]

    ax_scatter.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax_scatter.set_xlabel("Predicted Value ($y$)")
    ax_scatter.set_ylabel(r"$\Delta(x)$ (Correction Cost)")
    ax_scatter.legend(handles=legend_elements, loc='upper right', frameon=True)
    ax_scatter.grid(alpha=0.2)
    
    fig.suptitle(f"Unawareness Fairness Correction Transport Map (Sampled {len(y_u)} points)", y=0.95)
    
    #plt.tight_layout()
    plt.savefig("./results/generic_data_unaware_correction_line.png", dpi=300) 
    plt.show()

# unawareness plan
plot_fairness_plan(
    y_unfair = y_std, 
    y_fair = y_fair, 
    s_attr = S_test, 
    delta = ot_reg.delta_predict, 
    n_samples = 400
)

#Awareness as a special case of unawareness 
plot_fairness_plan(
    y_unfair = y_std, 
    y_fair = y_fair_aware, 
    s_attr = S_test, 
    delta = -S_test*4+6, 
    n_samples = 50
)

# %% 
# Some experiments on alpha_0
# Setup the Experiment Parameters
# base regressor : linear

n_points = 2000  # LOT of points for smooth histograms
alphas =  [0.15, 1.5, 3.0]  # From no separability to perfect separability
alphas_to_plot = alphas.copy()  # Specific alphas to visualize histograms for
n_runs = 1  # Nombre de répétitions (générations) par valeur d'alpha -> 1 just pour histogram
noise_scale = 0.3
histogram_data = {}

# Tracking metrics (Means and Stds)
mse_unfair_mean, mse_unfair_std = [], []
mse_fair_mean, mse_fair_std = [], []
w1_unfair_mean, w1_unfair_std = [], []
w1_fair_mean, w1_fair_std = [], []

print(f"Running experiment over alpha values with {n_runs} runs per alpha...")

# 2. Loop through different alpha (separability) values
for alpha in alphas:
    # Temporary lists for the current alpha
    temp_mse_unf, temp_mse_fair = [], []
    temp_w1_unf, temp_w1_fair = [], []
    temp_ks_unf, temp_ks_fair = [], [] 

    for run in range(n_runs):
        # Generate large dataset for smoothness (change seed per run for variance)
        X_exp, Y_exp, S_exp = generate_linear_data(
            n=n_points, alpha_0=alpha, alpha_1=1, x_scale=1, noise_scale=noise_scale, seed=run + int(alpha*100)
        )
        
        X_train_exp, X_test_exp, Y_train_exp, Y_test_exp, S_train_exp, S_test_exp = train_test_split(
            X_exp, Y_exp, S_exp, train_size=0.8, random_state=run
        )
        
        #training the unfair regressor

        std_reg_exp = LinearRegression().fit(X_train_exp, Y_train_exp)
        y_unfair_exp = std_reg_exp.predict(X_test_exp)
        
        
        try:
            #training OT Unaware Fair Regressor
            #ot_reg_exp = OTUnawareFairRegressor(base_regressor = std_reg_exp)
            ot_reg_exp = OTUnawareFairRegressor()
            ot_reg_exp.fit(X_train_exp, Y_train_exp, S_train_exp)
            y_fair_exp = ot_reg_exp.predict(X_test_exp, prediction="knn")
            delta_exp = ot_reg_exp.delta_predict
            
        except AssertionError:
            #If proxy collapses because alpha is too low (no separability)
            if run == 0:
                print(f"Alpha {alpha:.2f}: Proxy collapsed (no separability). Using unfair baseline.")
            y_fair_exp = y_unfair_exp.copy()
            delta_exp = np.random.randn(len(y_fair_exp)) #Dummy delta
            
        #Split condition based on delta
        mask_pos = (S_test_exp == 1)
        mask_neg = (S_test_exp == 2)
        
        #Ensure we don't calculate Wasserstein on empty arrays
        if sum(mask_pos) > 0 and sum(mask_neg) > 0:
            w1_unf = wasserstein_distance(y_unfair_exp[mask_pos], y_unfair_exp[mask_neg])
            w1_f = wasserstein_distance(y_fair_exp[mask_pos], y_fair_exp[mask_neg])
            ks_unf = ks_2samp(y_unfair_exp[mask_pos], y_unfair_exp[mask_neg]).statistic 
            ks_f = ks_2samp(y_fair_exp[mask_pos], y_fair_exp[mask_neg]).statistic 
            
        else:
            w1_unf, w1_f = 0.0, 0.0
        
        #Compute & Store Metrics for this run
        temp_mse_unf.append(mean_squared_error(Y_test_exp, y_unfair_exp))
        temp_mse_fair.append(mean_squared_error(Y_test_exp, y_fair_exp))
        temp_w1_unf.append(w1_unf)
        temp_w1_fair.append(w1_f)
        temp_ks_unf.append(ks_unf)
        temp_ks_fair.append(ks_f)
        
        #Save histogram data only for the first run of the requested alphas
        if run == 0 and any(np.isclose(alpha, a, atol=0.1) for a in alphas_to_plot) and len(histogram_data) < len(alphas_to_plot):
            histogram_data[alpha] = {
                'y_u': y_unfair_exp, 'y_f': y_fair_exp, 
                'mask_pos': mask_pos, 'mask_neg': mask_neg, 'w1_f': w1_f, 'ks_f': ks_f

            }
            
    mse_unfair_mean.append(np.mean(temp_mse_unf))
    mse_unfair_std.append(np.std(temp_mse_unf))
    
    mse_fair_mean.append(np.mean(temp_mse_fair))
    mse_fair_std.append(np.std(temp_mse_fair))
    
    w1_unfair_mean.append(np.mean(temp_w1_unf))
    w1_unfair_std.append(np.std(temp_w1_unf))
    
    w1_fair_mean.append(np.mean(temp_w1_fair))
    w1_fair_std.append(np.std(temp_w1_fair))


# %%
alphas = np.array(alphas)
mse_unfair_mean, mse_unfair_std = np.array(mse_unfair_mean), np.array(mse_unfair_std)
mse_fair_mean, mse_fair_std = np.array(mse_fair_mean), np.array(mse_fair_std)
w1_unfair_mean, w1_unfair_std = np.array(w1_unfair_mean), np.array(w1_unfair_std)
w1_fair_mean, w1_fair_std = np.array(w1_fair_mean), np.array(w1_fair_std)

print("Experiment complete. Plotting results...")

#Metrics (MSE and Wasserstein Distance) with Confidence Intervals
fig, axes = plt.subplots(1, 2, figsize=(10, 3))

# MSE Plot
axes[0].plot(alphas, mse_unfair_mean, marker='o', linestyle='--', color='gray', label='Unfair Regressor')
axes[0].fill_between(alphas, mse_unfair_mean - mse_unfair_std, mse_unfair_mean + mse_unfair_std, color='gray', alpha=0.2)

axes[0].plot(alphas, mse_fair_mean, marker='*', color='tab:red', markersize=10, label='Fair Regressor')
axes[0].fill_between(alphas, mse_fair_mean - mse_fair_std, mse_fair_mean + mse_fair_std, color='tab:red', alpha=0.2)

axes[0].set_title(r"Mean Squared Error vs Separability ($\alpha_0)")
axes[0].set_xlabel(r"Separability ($\alpha_0$)")
axes[0].set_ylabel("MSE")
axes[0].grid(True, linestyle=':', alpha=0.6)
axes[0].legend()

# Wasserstein Distance Plot 
axes[1].plot(alphas, w1_unfair_mean, marker='o', linestyle='--', color='gray', label='Unfair Regressor')
axes[1].fill_between(alphas, w1_unfair_mean - w1_unfair_std, w1_unfair_mean + w1_unfair_std, color='gray', alpha=0.2)

axes[1].plot(alphas, w1_fair_mean, marker='*', color='tab:blue', markersize=10, label='Fair Regressor')
axes[1].fill_between(alphas, w1_fair_mean - w1_fair_std, w1_fair_mean + w1_fair_std, color='tab:blue', alpha=0.2)

axes[1].axhline(0, color='black', linewidth=1, linestyle='-', alpha=0.5)
axes[1].set_title(r"Wasserstein Distance ($W_1$) conditioned on S")
axes[1].set_xlabel(r"Separability ($\alpha_0$)")
axes[1].set_ylabel(r"$W_1$ Distance")
axes[1].grid(True, linestyle=':', alpha=0.6)
axes[1].legend()

plt.tight_layout()
plt.show()


#Smooth histograms for specific alphas
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
    #plotting unfair Histograms
    ax_unf.hist(y_u[mask_pos], bins=bins, density=True, alpha=0.5, color=c_pos, label=r'S = 1')
    ax_unf.hist(y_u[mask_neg], bins=bins, density=True, alpha=0.5, color=c_neg, label=r'S = 2')
    ax_unf.set_title(r"Unfair Predictions ($\alpha_0 = %.1f$)" % alpha)
    ax_unf.set_ylabel("Density")
    if idx == 0 :
        
        ax_unf.legend(loc='upper right')
    ax_unf.grid(axis='y', alpha=0.3)
    
  
    #plotting fair histograms (barycenter)
    ax_fair.hist(y_f[mask_pos], bins=bins, density=True, alpha=0.5, color=c_pos, label=r'Fair | S = 1')
    ax_fair.hist(y_f[mask_neg], bins=bins, density=True, alpha=0.5, color=c_neg, label=r'Fair | S = 2')
    
    #adding an outline for the overall barycenter distribution
    ax_fair.hist(y_f, bins=bins, density=True, histtype='step', linewidth=2, color='black', linestyle='--', label='Overall Barycenter')
    
    ax_fair.set_title(r"Fair Predictions ($\alpha_0 = %.1f$) | $W_1 = %.4f$ | KS = %.4f" % (alpha, data['w1_f'], data['ks_f']))
    if idx == 0 :
        ax_fair.legend(loc='upper right')
    ax_fair.grid(axis='y', alpha=0.3)

plt.suptitle("Distributions Before and After Fairness Correction (linear base regressor)", fontsize=16, y=1.02)
plt.tight_layout()
plt.show()


# %%
