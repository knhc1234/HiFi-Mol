import os
import numpy as np
import pandas as pd

result_dir = './results/'

# Classification 
classification_datasets = ['bbbp', 'tox21', 'toxcast', 'hiv', 'muv', 'clintox', 'bace', 'sider']
cls_metric_col = 'Test_AUC_at_Best_Val'
cls_means = []

print("=" * 55)
print("[Classification] ROC-AUC")
print("=" * 55)
print(f"{'Dataset':<12} {'Mean':>10} {'Std':>10} {'N':>5}")
print("=" * 55)

for dataset in classification_datasets:
    csv_path = os.path.join(result_dir, f'HiFi-Mol_{dataset}.csv')

    if not os.path.exists(csv_path):
        print(f"{dataset:<12} {'File not found':>25}")
        continue

    df = pd.read_csv(csv_path)

    if cls_metric_col not in df.columns:
        print(f"{dataset:<12} {'Column not found':>25}")
        continue

    values = df[cls_metric_col].dropna().values
    mean = np.mean(values)
    std = np.std(values)
    cls_means.append(mean)

    print(f"{dataset:<12} {mean:>10.4f} {std:>10.4f} {len(values):>5}")

print("=" * 55)
if cls_means:
    print(f"{'Average':<12} {np.mean(cls_means):>10.4f}")
print("=" * 55)

# Regression
regression_datasets = ['esol', 'freesolv', 'lipophilicity']
reg_metric_col = 'Test_RMSE_at_Best_Val'
reg_means = []

print()
print("=" * 55)
print("[Regression] RMSE")
print("=" * 55)
print(f"{'Dataset':<12} {'Mean':>10} {'Std':>10} {'N':>5}")
print("=" * 55)

for dataset in regression_datasets:
    csv_path = os.path.join(result_dir, f'HiFi-Mol_{dataset}.csv')

    if not os.path.exists(csv_path):
        print(f"{dataset:<12} {'File not found':>10}")
        continue

    df = pd.read_csv(csv_path)

    if reg_metric_col not in df.columns:
        print(f"{dataset:<12} {'Column not found':>10}")
        continue

    values = df[reg_metric_col].dropna().values
    mean = np.mean(values)
    std = np.std(values)
    reg_means.append(mean)

    print(f"{dataset:<12} {mean:>10.4f} {std:>10.4f} {len(values):>5}")

print("=" * 55)
if reg_means:
    print(f"{'Average':<12} {np.mean(reg_means):>10.4f}")
print("=" * 55)
