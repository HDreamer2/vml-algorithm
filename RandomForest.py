from collections import Counter

import numpy as np
import pandas as pd
import random
from Constant import RANDOM_FOREST_GET_DATA

class RandomForest:
    def __init__(self, n_trees=10, max_depth=None, features_num=3):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.features_num = features_num
        self.trees = []
        self.selected_features = []

    def bootstrap_sample(self, X, y):
        n_samples = X.shape[0]
        indices = np.random.choice(n_samples, n_samples, replace=True)
        return X[indices], y[indices]

    def fit(self, X, y):
        n_features = X.shape[1]
        self.trees = []
        self.selected_features = []

        for _ in range(self.n_trees):
            # 随机选择特征
            features_indices = np.random.choice(n_features, self.features_num, replace=False)
            self.selected_features.append(features_indices)

            X_sample, y_sample = self.bootstrap_sample(X[:, features_indices], y)
            tree = DecisionTree(max_depth=self.max_depth)
            tree.fit(X_sample, y_sample)
            self.trees.append(tree)

    def predict(self, X):
        tree_preds = []
        for tree, features in zip(self.trees, self.selected_features):
            preds = tree.predict(X[:, features])
            tree_preds.append(preds)
            print(f"Predictions of tree {self.trees.index(tree) + 1}: {preds}")

        tree_preds = np.array(tree_preds)
        return [Counter(tree_preds[:, i]).most_common(1)[0][0] for i in range(X.shape[0])]

    def print_trees(self):
        for i, (tree, features) in enumerate(zip(self.trees, self.selected_features)):
            print(f"\nTree {i + 1}:")
            print(f"Features: {features}")
            print(tree.tree)

def RandomForestModel(data, l1, l2):
    X = data[l1].to_numpy()
    y = data[l2].to_numpy().flatten()
    tree = RandomForest(max_depth=3)
    tree.fit(X, y)
    transfer_data(tree.tree)

def convert_to_serializable(data):
    if isinstance(data, np.int64):
        return int(data)
    elif isinstance(data, np.float64):
        return float(data)
    elif isinstance(data, (list, tuple)):
        return [convert_to_serializable(i) for i in data]
    elif isinstance(data, dict):
        return {k: convert_to_serializable(v) for k, v in data.items()}
    else:
        return data

def transfer_data(tree):
    '''
    need to complete
    :param tree:
    :return:
    '''

    data = {
        'data': convert_to_serializable(tree)
    }
    response = requests.post(RANDOM_FOREST_GET_DATA, json=data)


# 示例用法
if __name__ == "__main__":
    data = pd.read_csv("C:/Users/admin/Desktop/random_data.csv")
    l1 = ["feature_0", "feature_1", "feature_2", "feature_3", "feature_4"]
    l2 = ["label"]

    X = data[l1].to_numpy()
    y = data[l2].to_numpy().flatten()

    # 获得随机森林
    forest = RandomForest(n_trees=10, max_depth=3, features_num=3)
    forest.fit(X, y)
    forest.print_trees()

    # 预测
    predictions = forest.predict(X)
    print("Predictions:", predictions)
