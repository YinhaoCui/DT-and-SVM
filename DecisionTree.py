import math
import numpy as np


class DecisionTreeClassifier(object):
    def __init__(self, max_depth):
        self.depth = 0
        self.max_depth = max_depth

    def entropy_func(self, c, n):
        value = -(c * 1.0 / n) * math.log(c * 1.0 / n, 2)
        return value

    def entropy_cal(self, c1, c2):
        if c1 == 0:
            return 0
        if c2 == 0:
            return 0
        value_1 = self.entropy_func(c1, c1 + c2)
        value_2 = self.entropy_func(c2, c1 + c2)
        return value_1 + value_2

    def entropy_of_one_division(self, division):
        s = 0
        classes = set(division)
        for c in classes:
            e = sum(division == c) * 1.0 / len(division) * self.entropy_cal(sum(division == c), sum(division != c))
            s = s + e
        return s, len(division)

    def get_entropy(self, y_predict, y_real):
        assert len(y_predict) == len(y_real)

        s_true, n_true = self.entropy_of_one_division(y_real[y_predict])
        s_false, n_false = self.entropy_of_one_division(y_real[~y_predict])
        value_1 = n_true * 1.0 / len(y_real) * s_true
        value_2 = n_false * 1.0 / len(y_real) * s_false
        return value_1 + value_2

    def fit(self, x, y, par_node={}, depth=0):
        if par_node is None:
            return None
        if len(y) == 0:
            return None
        if self.all_same(y):
            return {'val': y[0]}
        if depth >= self.max_depth:
            return None
        col, cutoff, entropy = self.find_best_split_of_all(x, y)  # find one split given an information gain
        y_left = y[x[:, col] < cutoff]
        y_right = y[x[:, col] >= cutoff]
        par_node = {'index_col': col, 'cutoff': cutoff, 'val': np.round(np.mean(y)),
                    'left': self.fit(x[x[:, col] < cutoff], y_left, {}, depth + 1),
                    'right': self.fit(x[x[:, col] >= cutoff], y_right, {}, depth + 1)}
        self.depth += 1
        self.trees = par_node
        return par_node


    def find_best_split_of_all(self, x, y):
        col = None
        min_entropy = 1
        cutoff = None
        for i, c in enumerate(x.T):
            entropy, cur_cutoff = self.find_best_split(c, y)
            if entropy == 0:
                return i, cur_cutoff, entropy
            elif entropy <= min_entropy:
                min_entropy = entropy
                col = i
                cutoff = cur_cutoff
        return col, cutoff, min_entropy


    def find_best_split(self, col, y):
        min_entropy = 10
        for value in set(col):
            y_predict = col < value
            my_entropy = self.get_entropy(y_predict, y)
            if my_entropy <= min_entropy:
                min_entropy = my_entropy
                cutoff = value
        return min_entropy, cutoff


    def all_same(self, items):
        return all(x == items[0] for x in items)


    def predict(self, x):
        results = np.array([0] * len(x))
        for i, c in enumerate(x):
            results[i] = self._get_prediction(c)
        return results


    def _get_prediction(self, row):
        cur_layer = self.trees
        while cur_layer.get('cutoff'):
            if row[cur_layer['index_col']] < cur_layer['cutoff']:
                cur_layer = cur_layer['left']
            else:
                cur_layer = cur_layer['right']
        else:
            return cur_layer.get('val')
