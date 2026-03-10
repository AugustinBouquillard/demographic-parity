import numpy as np
import pandas as pd
import os
import sys
import itertools
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.base import clone
from ucimlrepo import fetch_ucirepo 


from OTAwareFairRegressor import OTAwareFairRegressor
from OTUnawareFairRegressor import OTUnawareFairRegressor


current_dir = os.getcwd()
folder_path = os.path.abspath(os.path.join(current_dir, 'unaware-fair-reg-3rd-method'))
if folder_path not in sys.path:
    sys.path.insert(0, folder_path)
from FairReg import FairReg


class MultiClassOTUnawareFairRegressor:
    """
    naive One-vs-Rest extension for the binary OTUnawareFairRegressor.
    we train K binary fair regressors (one for each class vs the rest) and average predictions.
    """
    def __init__(self, base_regressor=None):
        self.base_regressor = base_regressor
        self.models = {}
        self.classes = None

    def fit(self, X, y, S):
        self.classes = np.unique(S)
        for c in self.classes:
            print(f"      Fitting Unaware OvR for Class {c}...")
            S_binary = np.where(S == c, 1, 0)
            
            model = OTUnawareFairRegressor(
                base_regressor=clone(self.base_regressor) if self.base_regressor is not None else None
            )
            model.fit(X, y, S_binary)
            self.models[c] = model
        return self

    def predict(self, X, prediction="knn"):
        preds = np.zeros((X.shape[0], len(self.classes)))
        for idx, c in enumerate(self.classes):
            preds[:, idx] = self.models[c].predict(X, prediction=prediction).flatten()
        return np.mean(preds, axis=1)

#data preparation utils
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

#evauaion 
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

def plot_multiclass_histograms(predictions_dict, S_test_arr, y_test_arr):
    groups = np.unique(S_test_arr)
    #mapping based on the integers assigned in get_communities_data_multiclass
    group_names = {0: 'Black', 1: 'White', 2: 'Asian', 3: 'Hispanic'}
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for idx, (name, preds) in enumerate(predictions_dict.items()):
        ax = axes[idx]
        
        mse = mean_squared_error(y_test_arr, preds)
        max_w1, max_ks = calculate_max_fairness_violation(preds, S_test_arr)
        
        #one histogram for each sensitive group
        for g in groups:
            yp_g = preds[S_test_arr == g]
            if len(yp_g) > 0:
                ax.hist(yp_g, bins=25, density=True, alpha=0.5, 
                        label=f'{group_names[g]}', edgecolor='white')
        
        ax.set_title(f"{name}\nMSE: {mse:.4f} | Max $W_1$: {max_w1:.4f} | Max KS: {max_ks:.4f}")
        ax.set_xlabel("Predicted Violent Crimes Per Pop")
        if idx % 2 == 0:
            ax.set_ylabel("Density")
        ax.legend(title="Majority Race")
        ax.grid(axis='y', linestyle=':', alpha=0.6)
        
    plt.suptitle("conditional output distributions by majority community (Communities & Crime)", fontsize=18, y=1.02)
    plt.tight_layout()
    plt.show()


def main():
    X, S, y = get_communities_data_multiclass()
    p = get_frequencies(S)
    
    TRAIN_SIZE, UNLAB_SIZE, TEST_SIZE = 0.4, 0.4, 0.2
    X_train, X_, S_train, S_, y_train, y_ = train_test_split(
        X, S, y, train_size=TRAIN_SIZE, stratify=S, random_state=42
    )
    X_unlab, X_test, S_unlab, S_test, y_unlab, y_test = train_test_split(
        X_, S_, y_, test_size=TEST_SIZE/(1-TRAIN_SIZE), stratify=S_, random_state=42
    )
    
    scaler = StandardScaler()
    X_train_arr = scaler.fit_transform(X_train)
    X_unlab_arr = scaler.transform(X_unlab)
    X_test_arr  = scaler.transform(X_test)
    
    y_train_arr, S_train_arr = np.array(y_train).flatten(), np.array(S_train).flatten()
    y_test_arr, S_test_arr = np.array(y_test).flatten(), np.array(S_test).flatten()

    clf_s = LogisticRegression(solver='lbfgs', max_iter=2000)
    clf_s.fit(X_train_arr, S_train_arr)
    S_test_est = clf_s.predict(X_test_arr).flatten()
 
    base_rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)


    print("\nTraining Base Model (Unfair Random Forest)...")
    reg = clone(base_rf)
    reg.fit(X_train_arr, y_train_arr)
    y_pred_base = reg.predict(X_test_arr).flatten()
    
    print("Training Taturyan et al....")
    fair_reg = FairReg(reg, clf_s, B=1, K=4, p=p, eps=[0.00001]*4, T=10000, keep_history=False)
    fair_reg.fit(X_unlab)
    y_pred_tat = fair_reg.predict(X_test).flatten()
    
    print("Training OT Aware Fair Regressor (Plug-in)...")
    ot_aware = OTAwareFairRegressor(base_estimator_model=clone(base_rf))
    ot_aware.fit(X_train_arr, y_train_arr, S_train_arr)
    y_pred_aware = ot_aware.predict(X_test_arr, S=S_test_est).flatten() 

    print("Training OT Unaware Fair Regressor (OvR Heuristic)...")
    ot_unaware_multi = MultiClassOTUnawareFairRegressor(base_regressor=clone(base_rf))
    ot_unaware_multi.fit(X_train_arr, y_train_arr, S_train_arr)
    y_pred_unaware = ot_unaware_multi.predict(X_test_arr, prediction="knn").flatten()

    print("\n" + "=" * 90)
    print(f"{'Model':<35} | {'MSE':<10} | {'Max W1':<10} | {'Max KS'}")
    print("=" * 90)
    
    predictions = {
        "Base model (unfair)": y_pred_base,
        "Taturyan et al.": y_pred_tat,
        "OT aware-derived": y_pred_aware,
        "OT unaware(OvR naive extension)": y_pred_unaware
    }
    
    for name, preds in predictions.items():
        mse = mean_squared_error(y_test_arr, preds)
        max_w1, max_ks = calculate_max_fairness_violation(preds, S_test_arr)
        print(f"{name:<35} | {mse:<10.4f} | {max_w1:<10.4f} | {max_ks:.4f}")
    print("=" * 90)
    
    print("\nPlotting multi-class prediction distributions...")
    plot_multiclass_histograms(predictions, S_test_arr, y_test_arr)
    

if __name__ == "__main__":
    main()