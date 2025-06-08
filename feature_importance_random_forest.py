import os
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.inspection import permutation_importance
from datetime import datetime

# Constants
CONTINUOUS_FEATURES = [
    'ft_start_addr', 'ft_length', 'cycles_since_btb_rec',
    'cycles_since_ibtb_rec', 'cycles_since_misfetch_rec',
    'cycles_since_mispred_rec', 'btb_miss_rate', 'ibtb_miss_rate',
    'misfetch_rate', 'mispred_rate',
]

CATEGORICAL_FEATURES = [
    'ft_ended_by', 'tage_comp_base', 'tage_comp_short',
    'tage_comp_long', 'tage_comp_loop', 'tage_comp_sc'
]

WORKLOAD_FEATHERS = {
    'clang': ['../icache_consumed_data_TEST/clang.feather'],
    'gcc': ['../icache_consumed_data_TEST/gcc.feather'],
    #'mysql': ['../icache_consumed_data_TESTmysql.feather'],
    #'mongodb': ['../icache_consumed_data_TEST/mongodb.feather'],
    #'postgres': ['../icache_consumed_data_TEST/postgres.feather'],
    #'verilator': ['../icache_consumed_data_TEST/verilator.feather'],
    #'xgboost': ['../icache_consumed_data_TEST/xgboost.feather'],
}

RANDOM_STATE = 42
TEST_SIZE = 0.25
N_ESTIMATORS = 100
N_JOBS = 3

OUTPUT_DIRS = {
    'gini': Path('Gini_Importance'),
    'permutation': Path('mean_decrease_accuracy')
}


def ensure_dirs(dirs):
    for directory in dirs.values():
        directory.mkdir(parents=True, exist_ok=True)


def load_all_data(feather_dict):
    """Load and concatenate all workloads into a single DataFrame."""
    dfs = [pd.read_feather(fp) for paths in feather_dict.values() for fp in paths]
    return pd.concat(dfs, ignore_index=True)


def load_data_for_workload(paths):
    """Load and concatenate a single workload's DataFrame."""
    dfs = [pd.read_feather(fp) for fp in paths]
    return pd.concat(dfs, ignore_index=True)


def train_random_forest(X_train, y_train):
    clf = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    )
    clf.fit(X_train, y_train)
    return clf


def plot_importances(names, importances, title, xlabel, output_path):
    plt.figure(figsize=(8, 6))
    idx = np.argsort(importances)
    plt.barh(np.arange(len(names)), importances[idx], align='center')
    plt.yticks(np.arange(len(names)), np.array(names)[idx])
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def process_workload(name, df):
    print(f"\n=== Processing workload: {name} ===")
    features = CONTINUOUS_FEATURES + CATEGORICAL_FEATURES
    X = df[features]
    y = df['off_path']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    print(f'Beginning random forest training for {name} workload at {datetime.now()}...')
    clf = train_random_forest(X_train, y_train)
    print(f'Finished random forest training for {name} workload at {datetime.now()}.')

    # Gini Importance
    gini_importances = clf.feature_importances_
    gini_df = pd.DataFrame({
        'Feature': features,
        'Gini Importance': gini_importances
    }).sort_values('Gini Importance', ascending=False).reset_index(drop=True)
    print(gini_df)
    plot_importances(
        features, gini_importances,
        'Feature Importance - Gini Importance',
        'Gini Importance',
        OUTPUT_DIRS['gini'] / f"{name}.png"
    )
    print(f'Caclulating permulation importance for {name} workload at {datetime.now()}...')
    # Permutation Importance
    perm_result = permutation_importance(
        clf, X_test, y_test,
        n_repeats=10,
        random_state=RANDOM_STATE,
        n_jobs=N_JOBS
    )
    perm_importances = perm_result.importances_mean
    perm_df = pd.DataFrame({
        'Feature': features,
        'Decrease in Accuracy': perm_importances
    }).sort_values('Decrease in Accuracy', ascending=False).reset_index(drop=True)
    print(perm_df)
    plot_importances(
        features, perm_importances,
        'Feature Importance - Mean Decrease Accuracy',
        'Mean Decrease Accuracy',
        OUTPUT_DIRS['permutation'] / f"{name}.png"
    )


def main():
    ensure_dirs(OUTPUT_DIRS)

    # Individual workloads
    for workload, paths in WORKLOAD_FEATHERS.items():
        print(workload)
        df = load_data_for_workload(paths)
        process_workload(workload, df)

    # Combined 'all' workloads
    df_all = load_all_data(WORKLOAD_FEATHERS)
    print('all')
    process_workload('all', df_all)


if __name__ == '__main__':
    main()