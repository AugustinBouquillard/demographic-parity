#%%

import numpy as np

import ot 
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from scipy.stats import ks_2samp
from sklearn.model_selection import train_test_split
from OTUnawareFairRegressor import OTUnawareFairRegressor 
from data_extraction_script import read_adult_dataset


# %%
# Generate Data (X depends on S)
def generate_linear_data(n , alpha_0, alpha_1, p = 0.5, x_scale = 1, noise_scale = 1, seed = 42):
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

    X = np.random.normal(0, x_scale, n) + alpha_0 * S
    X = X.reshape(-1, 1)

    # Y depends on X and S 
    Y = alpha_1*S + 0.5 * X.flatten() + np.random.normal(0, noise_scale, n)
    Y = Y.reshape(-1, 1)
    return X, Y, S

# %%

# visualisation of generated dataset 
n = 600 
alpha_0 = 2
alpha_1 = 1 
x_scale = 1 
noise_scale = 1 
X, Y, S = generate_linear_data(n = 600, alpha_0 = alpha_0, alpha_1 = alpha_1, x_scale = x_scale, noise_scale = noise_scale)

# train-test split
X_train, X_test, Y_train, Y_test, S_train, S_test = train_test_split(X, Y, S, train_size = 0.7)

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


# dataset
plt.scatter(X.flatten(), Y.flatten(),c = S, cmap='tab10', vmax=9,alpha=0.5)

# regression line
X_array = np.linspace( - 3*x_scale + alpha_0 * 1, 3*x_scale + alpha_0 * 2, 1000).reshape(-1, 1)
plt.plot(X_array, std_reg.predict(X_array), label = "unfair regressor")
plt.plot(X_array, std_reg_maj.predict(X_array), label = "regressor for S = 1")
plt.plot(X_array, std_reg_min.predict(X_array), label = "regressor for S = 2")
plt.legend()
plt.show()


# %%
