import numpy as np
import pandas as pd
import os
import sys
import itertools
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from scipy.stats import wasserstein_distance, ks_2samp
from ucimlrepo import fetch_ucirepo 

# Import Custom Models
from OTAwareFairRegressor import OTAwareFairRegressor
from OTUnawareFairRegressor import OTUnawareFairRegressor

# Ensure FairReg can be imported from the unaware-fair-reg directory
current_dir = os.getcwd()
folder_path = os.path.abspath(os.path.join(current_dir, 'unaware-fair-reg-3rd-method'))
if folder_path not in sys.path:
    sys.path.insert(0, folder_path)
from FairReg import FairReg

# --- One-vs-Rest Wrapper for the Unaware Method ---
class MultiClassOTUnawareFairRegressor:
    """
    Heuristic One-vs-Rest extension for the binary OTUnawareFairRegressor.
    Trains K binary fair regressors (one for each class vs the rest) and averages predictions.
    """
    def __init__(self):
        self.models = {}
        self.classes = None

    def fit(self, X, y, S):
        self.classes = np.unique(S)
        for c in self.classes:
            print(f"      Fitting Unaware OvR for Class {c}...")
            # Create binary sensitive attribute: 1 if class c, 0 otherwise
            S_binary = np.where(S == c, 1, 0)
            
            # Initialize and fit a standard binary OT Unaware Regressor
            model = OTUnawareFairRegressor()
            model.fit(X, y, S_binary)
            self.models[c] = model
        return self

    def predict(self, X, prediction="knn"):
        # Gather predictions from all binary models
        preds = np.zeros((X.shape[0], len(self.classes)))
        for idx, c in enumerate(self.classes):
            preds[:, idx] = self.models[c].predict(X, prediction=prediction).flatten()
        
        # Average the predictions across all One-vs-Rest models
        return np.mean(preds, axis=1)


# --- Data Preparation ---
def get_frequencies(S):
    p = []
    for p_s in sorted(S.value_counts(normalize=True).sort_index()):
        p.append(p_s)
    return p 

def get_communities_data_multiclass(as_df=False):
    communities_and_crime = fetch_ucirepo(id=183)
    df = communities_and_crime.data.original
    df = df.fillna(0)

    cols_to_drop = ['communityname', 'state', 'county', 'community', 'fold']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns], errors='ignore')

    sens_attrs = ['racepctblack', 'racePctWhite', 'racePctAsian', 'racePctHisp']
    df['race'] = df[sens_attrs].idxmax(axis=1) 
    df = df.drop(columns=sens_attrs)

    df = df.drop(df[df['ViolentCrimesPerPop']==0].index)
    y = df['ViolentCrimesPerPop']
    df = df.drop('ViolentCrimesPerPop', axis=1)

    # 4 classes (0: Black, 1: White, 2: Asian, 3: Hispanic)
    mapping = {'racepctblack': 0, 'racePctWhite': 1, 'racePctAsian': 2, 'racePctHisp': 3} 
    S = df['race'].map(mapping) 
    df = df.drop('race', axis=1)
    
    X_crime = df.replace('?', np.nan).apply(pd.to_numeric, errors='coerce')
    imputer = SimpleImputer(strategy='mean')
    X_crime_clean = pd.DataFrame(imputer.fit_transform(X_crime), columns=X_crime.columns)
    
    if as_df:
        return X_crime_clean, S, y
    else:
        return X_crime_clean.to_numpy(), S, y

# --- Evaluation Helpers ---
def calculate_max_fairness_violation(preds, S_test_arr):
    groups = np.unique(S_test_arr)
    max_w1, max_ks = 0, 0
    
    for g1, g2 in itertools.combinations(groups, 2):
        yp_g1 = preds[S_test_arr == g1]
        yp_g2 = preds[S_test_arr == g2]
        
        if len(yp_g1) == 0 or len(yp_g2) == 0:
            continue
            
        w1 = wasserstein_distance(yp_g1, yp_g2)
        ks = ks_2samp(yp_g1, yp_g2).statistic
        
        if w1 > max_w1: max_w1 = w1
        if ks > max_ks: max_ks = ks
            
    return max_w1, max_ks

# --- Main Execution ---
def main():
    print("Loading Communities & Crime data (Multi-class S)...")
    X, S, y = get_communities_data_multiclass()
    p = get_frequencies(S)
    
    TRAIN_SIZE, UNLAB_SIZE, TEST_SIZE = 0.4, 0.4, 0.2
    X_train, X_, S_train, S_, y_train, y_ = train_test_split(
        X, S, y, train_size=TRAIN_SIZE, stratify=S, random_state=42
    )
    X_unlab, X_test, S_unlab, S_test, y_unlab, y_test = train_test_split(
        X_, S_, y_, test_size=TEST_SIZE/(1-TRAIN_SIZE), stratify=S_, random_state=42
    )
    
    X_train_arr, y_train_arr, S_train_arr = np.array(X_train), np.array(y_train).flatten(), np.array(S_train).flatten()
    X_unlab_arr = np.array(X_unlab)
    X_test_arr, y_test_arr, S_test_arr = np.array(X_test), np.array(y_test).flatten(), np.array(S_test).flatten()
    
    print("\nTraining Multi-Class Sensitive Attribute Estimator (for Aware Plug-in)...")
    clf_s = LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=2000)
    clf_s.fit(X_train_arr, S_train_arr)
    S_test_est = clf_s.predict(X_test_arr).flatten()
    
    # --- Models ---
    print("\nTraining Base Model (Unfair)...")
    reg = LinearRegression()
    reg.fit(X_train_arr, y_train_arr)
    y_pred_base = reg.predict(X_test_arr).flatten()
    
    print("Training Taturyan et al. (Minimax)...")
    fair_reg = FairReg(reg, clf_s, B=1, K=4, p=p, eps=[0.00001]*4, T=100000, keep_history=False)
    fair_reg.fit(X_unlab_arr)
    y_pred_tat = fair_reg.predict(X_test_arr).flatten()
    
    print("Training OT Aware Fair Regressor (Plug-in)...")
    ot_aware = OTAwareFairRegressor(base_estimator_model=LinearRegression())
    ot_aware.fit(X_train_arr, y_train_arr, S_train_arr)
    y_pred_aware = ot_aware.predict(X_test_arr, S=S_test_est).flatten() 

    print("Training OT Unaware Fair Regressor (OvR Heuristic)...")
    ot_unaware_multi = MultiClassOTUnawareFairRegressor()
    ot_unaware_multi.fit(X_train_arr, y_train_arr, S_train_arr)
    y_pred_unaware = ot_unaware_multi.predict(X_test_arr, prediction="knn").flatten()

    # --- Evaluation ---
    print("\n" + "=" * 90)
    print(f"{'Model':<35} | {'MSE':<10} | {'Max W1':<10} | {'Max KS'}")
    print("=" * 90)
    
    predictions = {
        "Base Model (Unfair)": y_pred_base,
        "Taturyan et al. (Minimax)": y_pred_tat,
        "OT Aware (Estimated S Plug-in)": y_pred_aware,
        "OT Unaware (OvR Heuristic)": y_pred_unaware
    }
    
    for name, preds in predictions.items():
        mse = mean_squared_error(y_test_arr, preds)
        max_w1, max_ks = calculate_max_fairness_violation(preds, S_test_arr)
        print(f"{name:<35} | {mse:<10.4f} | {max_w1:<10.4f} | {max_ks:.4f}")
    print("=" * 90)

if __name__ == "__main__":
    main()