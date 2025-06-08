import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

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

def tree_to_code(tree, feature_names, tree_id, forest_name):
    """Generates C++ if-else code for a single Decision Tree."""
    tree_ = tree.tree_

    def recurse(node, depth):
        indent = "    " * depth
        if tree_.feature[node] != -2:  # Not a leaf node
            name = feature_names[tree_.feature[node]]
            threshold = tree_.threshold[node]

            print(f"{indent}if ({name} <= {threshold:.6f}) {{")
            recurse(tree_.children_left[node], depth + 1)
            print(f"{indent}}} else {{")
            recurse(tree_.children_right[node], depth + 1)
            print(f"{indent}}}")
        else:  # Leaf node
            class_probabilities = tree_.value[node][0]
            predicted_class = np.argmax(class_probabilities)
            print(f"{indent}return {predicted_class};")

    print(f"int predict_{forest_name}_tree_{tree_id}(float " + ", float ".join(feature_names) + ") {")
    recurse(0, 1)
    print("}")

def forest_to_code(forest, feature_names, forest_name):
    """Generates C++ code for a full Random Forest model."""
    
    num_trees = len(forest.estimators_)
    
    print("// C++ Code for Random Forest Inference\n")
    
    # Generate tree functions
    for i, tree in enumerate(forest.estimators_):
        tree_to_code(tree, feature_names, i, forest_name)
        print("\n")

    # Generate voting function
    print("int predict_" + forest_name + "_forest(float " + ", float ".join(feature_names) + ") {")
    print("    int votes[{}] = {{0}};".format(num_trees))
    
    for i in range(num_trees):
        print(f"    votes[{i}] = predict_{forest_name}_tree_{i}(" + ", ".join(feature_names) + ");")
    
    print("\n    // Majority voting")
    print("    int class_counts[2] = {0};")  # Adjust based on number of classes
    print("    for (int i = 0; i < {}; i++) class_counts[votes[i]]++;".format(num_trees))
    print("    int best_class = 0;")
    print("    for (int i = 1; i < 2; i++) { if (class_counts[i] > class_counts[best_class]) best_class = i; }")
    print("    return best_class;")
    print("}")

for clf_path in [
    'max_depth_3_n_estimators_1_label_str_off_path/all_random_forest.pickle',
    'max_depth_3_n_estimators_10_label_str_off_path/all_random_forest.pickle',
    'max_depth_3_n_estimators_20_label_str_off_path/all_random_forest.pickle',
    'max_depth_5_n_estimators_1_label_str_off_path/all_random_forest.pickle',
    'max_depth_5_n_estimators_10_label_str_off_path/all_random_forest.pickle',
    'max_depth_5_n_estimators_20_label_str_off_path/all_random_forest.pickle',
    'max_depth_10_n_estimators_1_label_str_off_path/all_random_forest.pickle',
    'max_depth_10_n_estimators_10_label_str_off_path/all_random_forest.pickle',
    'max_depth_10_n_estimators_20_label_str_off_path/all_random_forest.pickle',
]:
    with open("max_depth_3_n_estimators_10_label_str_off_path/all_random_forest.pickle", "rb") as f:
        clf = pickle.load(f)

    # Generate C++ code
    forest_to_code(clf, continuous_features + categorical_features, clf_path.rstrip('/all_random_forest.pickle'))
