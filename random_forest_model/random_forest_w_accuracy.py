import pandas as pd
import numpy as np
import statistics
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt 
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd
import shap
from sklearn.metrics import accuracy_score
import numpy as np
import pickle
import datetime
from CustomAccuracyMetrics import get_cycle_penalty_max_offpath
from itertools import product
import json

n_estimators_l = [20, 10, 1]
max_depth_l = [3, 5, 10]
label_str_l = ['off_path_reason', 'off_path']

random_state = 42
window_size = 1
stride = 1


print(datetime.datetime.now())

continuous_features = [
    'ft_start_addr',
    'ft_length',
    'cycles_since_btb_rec',
    'cycles_since_ibtb_rec',
    'cycles_since_misfetch_rec',
    'cycles_since_mispred_rec',
    'btb_miss_rate',
    'ibtb_miss_rate',
    'misfetch_rate',
    'mispred_rate',
    # I'm sure these would be much more useful with execution driven :(((
    # 'dcache_miss_rate',
    # 'icache_miss_rate',
    # 'mlc_miss_rate',
    # 'l1_rate',
]

categorical_features = [
    'ft_ended_by',
    'tage_comp_base',
    'tage_comp_short',
    'tage_comp_long',
    'tage_comp_loop',
    'tage_comp_sc'
]

eval_columns = [
    'workload',
    'prediction',
    'label',
    'off_path_cycles',
    'cycle_penalty',
    'length_off_path',
    'window_id',
    'off_path',
    'off_path_reason'
]

# import data
workload_feathers = {
    'clang' : ['../icache_consumed_data/clang.feather'],
    'gcc' : ['../icache_consumed_data/gcc.feather',],
    'mysql' : ['../icache_consumed_data/mysql.feather',],
    'mongodb' : ['../icache_consumed_data/mongodb.feather',],
    'postgres' : ['../icache_consumed_data/postgres.feather',],
    'verilator' : ['../icache_consumed_data/verilator.feather',],
    'xgboost' : ['../icache_consumed_data/xgboost.feather',]
}

workloads = [
    'all',
    'clang',
    'gcc',
    'mysql',
    'mongodb',
    'postgres',
    'verilator',
    'xgboost'
]

cycles = {
    'all' : 584101761.0,
    'clang' : 9566632.0,
    'gcc' : 11505414.0,
    'mysql' : 10004884.0,
    'mongodb' : 12326291.0,
    'postgres' : 7771614.0,
    'verilator' : 35282687.0,
    'xgboost' : 497644239.0,
}

# add extra columns, effectively creating a sliding window in reach row.
def add_shifted_columns(df, window_size, columns, stride=1) -> str:
    window_features = []
    for column in columns:
        for i in range(0, window_size, stride):
            df[f'{column}_{i}'] = df[column].shift()
            window_features.append(f'{column}_{i}')
    df.dropna(inplace=True)
    return window_features

f = open('../icache_consumed_chunked_data_raw/train_test_files.json')
train_test_dict = json.load(f)
train_df = pd.concat([pd.read_feather(f'../icache_consumed_chunked_data_raw/{file}.feather') for file in train_test_dict['train']])
test_df = pd.concat([pd.read_feather(f'../icache_consumed_chunked_data_raw/{file}.feather') for file in train_test_dict['test']])
window_features = continuous_features + categorical_features

results_df = pd.DataFrame(columns=['max_depth', 'n_estimators', 'label', 'workload', 'regular_accuracy', 'custom_accuracy', 'slowdown', 'energy_savings'])
for config in product(max_depth_l, n_estimators_l, label_str_l):
    max_depth, n_estimators, label_str = config
    test_dir = f'max_depth_{max_depth}_n_estimators_{n_estimators}_label_str_{label_str}'
    print(f'n_estimators = {n_estimators}')
    print(f'random_state = {random_state}')
    print(f'max_depth: {max_depth}')
    print(f'window_size: {window_size}')
    print(f'stride: {stride}')

    if not os.path.exists(test_dir):
        os.mkdir(test_dir)

    train_df[label_str] = train_df[label_str].apply(lambda x: 1 if x > 0 else 0)
    test_df[label_str] = test_df[label_str].apply(lambda x: 1 if x > 0 else 0)

    feature_names = window_features
    X_train = train_df[window_features]
    y_train = train_df[label_str]

    X_test = test_df[window_features]
    y_test = test_df[label_str]

    # weights
    weights = train_df['cycle_penalty'].to_numpy()
    weights = np.where(weights < 50, weights + 1, 51)

    clf = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state, n_jobs=3, max_depth=max_depth)
    clf.fit(X_train, y_train, sample_weight=weights)

    predictions = clf.predict(X_train)
    initial_accuracy = accuracy_score(y_train, clf.predict(X_train)) # initial accuracy
    print("Train Accuracy: ", initial_accuracy)

    predictions = clf.predict(X_test)
    initial_accuracy = accuracy_score(y_test, clf.predict(X_test)) # initial accuracy
    print("Test Accuracy: ", initial_accuracy)

    df_out = test_df.copy()
    df_out['prediction'] = predictions
    # THIS IS INTENTIONAL
    df_out['label'] = df_out['off_path']

    df_out = df_out.sort_index()[eval_columns]
    df_out.to_feather(f'{test_dir}/all_random_forest.feather')

    print(datetime.datetime.now())
    print('- Standard Accuracies -')
    for workload in workloads:
        if(workload == 'all'):
            y_preds = df_out['prediction']
            y_labels = df_out[label_str]
            accuracy = accuracy_score(y_labels.to_numpy(), y_preds.to_numpy())
            print(f'{workload}: ', accuracy)
        else:
            y_preds = df_out[df_out['workload'] == workload]['prediction']
            y_labels = df_out[df_out['workload'] == workload][label_str]
            accuracy = accuracy_score(y_labels, y_preds)
            print(f'{workload}: ', accuracy)
        results_df.loc[len(results_df)] = [max_depth, n_estimators, label_str, workload, accuracy, None, None, None]

    print('- Custom Accuracies -')
    for workload in workloads:
        if workload == 'all':
            total_penalty, early_penalty, late_penalty, off_path_cycles, n_mispreds = get_cycle_penalty_max_offpath(df_out, workload)
            print(f'{workload}: {total_penalty / off_path_cycles}')
        else:
            total_penalty, early_penalty, late_penalty, off_path_cycles, n_mispreds = get_cycle_penalty_max_offpath(df_out[df_out['workload'] == workload], workload)
            print(f'{workload}: {total_penalty / off_path_cycles}')
        results_df.loc[len(results_df)] = [max_depth, n_estimators, label_str, workload, None, (1 - (total_penalty / off_path_cycles)), early_penalty / cycles[workload], (off_path_cycles - late_penalty) / cycles[workload]]

    print(datetime.datetime.now())

results_df.to_csv(f'forest_results_{datetime.datetime.now()}.csv', index=False)