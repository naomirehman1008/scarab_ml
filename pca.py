# Note: I don't think PCA is right for this data, it assumes linear relationships and normal distribution. But it looks cool.

import pandas as pd
from sklearn.feature_selection import f_classif, chi2
import numpy as np
import statistics
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import matplotlib.pyplot as plt
import sklearn.decomposition
from mpl_toolkits.mplot3d import Axes3D 
import matplotlib.pyplot as plt 
import os

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
    'dcache_miss_rate',
    'icache_miss_rate',
    'mlc_miss_rate',
    'l1_rate',
]

categorical_features = [
    'ft_ended_by',
    'tage_comp_base',
    'tage_comp_short',
    'tage_comp_long',
    'tage_comp_loop',
    'tage_comp_sc'
]

# import data
workload_feathers = {
    'clang' : ['icache_consumed_data/clang.feather'],
    'gcc' : ['icache_consumed_data/gcc.feather',],
    'mysql' : ['icache_consumed_data/mysql.feather',],
    'mongodb' : ['icache_consumed_data/mongodb.feather',],
    'postgres' : ['icache_consumed_data/postgres.feather',],
    'verilator' : ['icache_consumed_data/verilator.feather',],
    'xgboost' : ['icache_consumed_data/xgboost.feather',]
}


def plot_pca(df_to_pca, scaler_t, name):
    print(name)
    pca = sklearn.decomposition.PCA(n_components=3)
    if scaler_t == 'minmax':
        scaler = sklearn.preprocessing.MinMaxScaler()
    elif scaler_t == 'robust':
        scaler = sklearn.preprocessing.RobustScaler()
    elif scaler_t == 'standard':
        scaler = sklearn.preprocessing.StandardScaler()
    df_scaled = scaler.fit_transform(df_to_pca[continuous_features])
    pca_data = pca.fit_transform(df_scaled)
    print(pca.components_)
    pdf = pd.DataFrame(pca_data, columns=['pc_0', 'pc_1', 'pc_2'])
    plt.clf()
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d') 
    colors = ['#FF0000', '#0000FF', '#00FF00', '#FFA500', '#800080', '#00FFFF', '#FF00FF']
    point_colors = [colors[x] for x in df_to_pca['off_path'].values]

    # Plot the points with colors
    sc = ax.scatter(pdf['pc_0'],
                    pdf['pc_1'],
                    pdf['pc_2'],
                    c=point_colors,
                    marker='.'
                    ) 
    ax.set_xlabel('Component 0')
    ax.set_ylabel('Component 1')
    ax.set_zlabel('Component 2')
    ax.set_title(name)

    ax.legend()
    plt.savefig(os.path.join(os.getcwd(), 'feature_selection', scaler_t, name + '.png'), dpi =300 )

for scaler_t in ['minmax', 'robust', 'standard']:
    print(scaler_t)
    for workload in workload_feathers.keys():
        df = pd.read_feather(workload_feathers[workload][0])
        plot_pca(df, scaler_t, workload)

