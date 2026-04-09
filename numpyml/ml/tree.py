import numpy as np


class DecisionTree:
    def __init__(self, max_depth=10, min_samples_split=2, min_samples_leaf=1,
                 criterion='gini', task='classification'):
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.criterion = criterion
        self.task = task
        self.tree = None

    def _gini(self, y):
        classes, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return 1 - np.sum(p ** 2)

    def _entropy(self, y):
        classes, counts = np.unique(y, return_counts=True)
        p = counts / len(y)
        return -np.sum(p * np.log2(p + 1e-10))

    def _mse(self, y):
        return np.var(y) if len(y) > 0 else 0

    def _impurity(self, y):
        if self.task == 'regression':
            return self._mse(y)
        if self.criterion == 'gini':
            return self._gini(y)
        return self._entropy(y)

    def _best_split(self, X, y):
        best_gain = -1
        best_feat = None
        best_thresh = None
        n = len(y)
        parent_impurity = self._impurity(y)
        for feat in range(X.shape[1]):
            thresholds = np.unique(X[:, feat])
            if len(thresholds) > 20:
                thresholds = np.percentile(X[:, feat], np.linspace(0, 100, 20))
            for thresh in thresholds:
                left = y[X[:, feat] <= thresh]
                right = y[X[:, feat] > thresh]
                if len(left) < self.min_samples_leaf or len(right) < self.min_samples_leaf:
                    continue
                gain = parent_impurity - (len(left) / n * self._impurity(left) + len(right) / n * self._impurity(right))
                if gain > best_gain:
                    best_gain = gain
                    best_feat = feat
                    best_thresh = thresh
        return best_feat, best_thresh, best_gain

    def _build(self, X, y, depth):
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            if self.task == 'classification':
                classes, counts = np.unique(y, return_counts=True)
                return {'leaf': True, 'value': classes[np.argmax(counts)], 'proba': counts / len(y), 'classes': classes}
            return {'leaf': True, 'value': np.mean(y)}
        feat, thresh, gain = self._best_split(X, y)
        if feat is None or gain <= 0:
            if self.task == 'classification':
                classes, counts = np.unique(y, return_counts=True)
                return {'leaf': True, 'value': classes[np.argmax(counts)], 'proba': counts / len(y), 'classes': classes}
            return {'leaf': True, 'value': np.mean(y)}
        left_mask = X[:, feat] <= thresh
        return {
            'leaf': False, 'feature': feat, 'threshold': thresh,
            'left': self._build(X[left_mask], y[left_mask], depth + 1),
            'right': self._build(X[~left_mask], y[~left_mask], depth + 1),
        }

    def fit(self, X, y):
        self.n_classes = len(np.unique(y)) if self.task == 'classification' else None
        self.tree = self._build(X, y, 0)
        return self

    def _predict_one(self, x, node):
        if node['leaf']:
            return node['value']
        if x[node['feature']] <= node['threshold']:
            return self._predict_one(x, node['left'])
        return self._predict_one(x, node['right'])

    def _predict_proba_one(self, x, node):
        if node['leaf']:
            proba = np.zeros(self.n_classes)
            for c, p in zip(node['classes'], node['proba']):
                proba[c] = p
            return proba
        if x[node['feature']] <= node['threshold']:
            return self._predict_proba_one(x, node['left'])
        return self._predict_proba_one(x, node['right'])

    def predict(self, X):
        return np.array([self._predict_one(x, self.tree) for x in X])

    def predict_proba(self, X):
        return np.array([self._predict_proba_one(x, self.tree) for x in X])

    def score(self, X, y):
        preds = self.predict(X)
        if self.task == 'classification':
            return np.mean(preds == y)
        return 1 - np.sum((y - preds) ** 2) / np.sum((y - np.mean(y)) ** 2)


class RandomForest:
    def __init__(self, n_estimators=100, max_depth=10, min_samples_split=2,
                 max_features='sqrt', criterion='gini', task='classification',
                 bootstrap=True, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.max_features = max_features
        self.criterion = criterion
        self.task = task
        self.bootstrap = bootstrap
        self.random_state = random_state
        self.trees = []
        self.feature_indices = []

    def _get_n_features(self, n_total):
        if self.max_features == 'sqrt':
            return max(1, int(np.sqrt(n_total)))
        elif self.max_features == 'log2':
            return max(1, int(np.log2(n_total)))
        elif isinstance(self.max_features, float):
            return max(1, int(self.max_features * n_total))
        elif isinstance(self.max_features, int):
            return self.max_features
        return n_total

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        n_feat = self._get_n_features(n_features)
        self.trees = []
        self.feature_indices = []
        self.n_classes = len(np.unique(y)) if self.task == 'classification' else None
        for _ in range(self.n_estimators):
            if self.bootstrap:
                idx = rng.randint(0, n_samples, n_samples)
            else:
                idx = np.arange(n_samples)
            feat_idx = rng.choice(n_features, n_feat, replace=False)
            tree = DecisionTree(max_depth=self.max_depth, min_samples_split=self.min_samples_split,
                                criterion=self.criterion, task=self.task)
            tree.fit(X[idx][:, feat_idx], y[idx])
            self.trees.append(tree)
            self.feature_indices.append(feat_idx)
        return self

    def predict(self, X):
        if self.task == 'classification':
            preds = np.array([tree.predict(X[:, feat_idx])
                              for tree, feat_idx in zip(self.trees, self.feature_indices)])
            return np.apply_along_axis(lambda x: np.bincount(x.astype(int)).argmax(), 0, preds)
        preds = np.array([tree.predict(X[:, feat_idx])
                          for tree, feat_idx in zip(self.trees, self.feature_indices)])
        return np.mean(preds, axis=0)

    def predict_proba(self, X):
        probas = np.zeros((len(X), self.n_classes))
        for tree, feat_idx in zip(self.trees, self.feature_indices):
            probas += tree.predict_proba(X[:, feat_idx])
        return probas / self.n_estimators

    def score(self, X, y):
        preds = self.predict(X)
        if self.task == 'classification':
            return np.mean(preds == y)
        return 1 - np.sum((y - preds) ** 2) / np.sum((y - np.mean(y)) ** 2)

    def feature_importances(self, X, y):
        base_score = self.score(X, y)
        importances = np.zeros(X.shape[1])
        for feat in range(X.shape[1]):
            X_perm = X.copy()
            X_perm[:, feat] = np.random.permutation(X_perm[:, feat])
            perm_score = self.score(X_perm, y)
            importances[feat] = base_score - perm_score
        importances = np.maximum(importances, 0)
        total = importances.sum()
        if total > 0:
            importances /= total
        return importances


class GradientBoostedTrees:
    def __init__(self, n_estimators=100, max_depth=3, learning_rate=0.1,
                 subsample=1.0, task='classification', random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.task = task
        self.random_state = random_state
        self.trees = []

    def _softmax(self, z):
        e = np.exp(z - np.max(z, axis=1, keepdims=True))
        return e / np.sum(e, axis=1, keepdims=True)

    def fit(self, X, y):
        rng = np.random.RandomState(self.random_state)
        n = len(y)
        if self.task == 'classification':
            self.n_classes = len(np.unique(y))
            self.trees = [[] for _ in range(self.n_classes)]
            F = np.zeros((n, self.n_classes))
            class_counts = np.bincount(y, minlength=self.n_classes)
            F += np.log(class_counts / n + 1e-10)
            self._init_F = F[0].copy()
            for _ in range(self.n_estimators):
                proba = self._softmax(F)
                for c in range(self.n_classes):
                    residuals = (y == c).astype(float) - proba[:, c]
                    idx = rng.choice(n, int(n * self.subsample), replace=False) if self.subsample < 1.0 else np.arange(n)
                    tree = DecisionTree(max_depth=self.max_depth, task='regression')
                    tree.fit(X[idx], residuals[idx])
                    F[:, c] += self.learning_rate * tree.predict(X)
                    self.trees[c].append(tree)
        else:
            self.trees = []
            self._init_val = np.mean(y)
            F = np.full(n, self._init_val)
            for _ in range(self.n_estimators):
                residuals = y - F
                idx = rng.choice(n, int(n * self.subsample), replace=False) if self.subsample < 1.0 else np.arange(n)
                tree = DecisionTree(max_depth=self.max_depth, task='regression')
                tree.fit(X[idx], residuals[idx])
                F += self.learning_rate * tree.predict(X)
                self.trees.append(tree)
        return self

    def predict(self, X):
        if self.task == 'classification':
            F = np.tile(self._init_F, (len(X), 1))
            for c in range(self.n_classes):
                for tree in self.trees[c]:
                    F[:, c] += self.learning_rate * tree.predict(X)
            return np.argmax(F, axis=1)
        F = np.full(len(X), self._init_val)
        for tree in self.trees:
            F += self.learning_rate * tree.predict(X)
        return F

    def predict_proba(self, X):
        F = np.tile(self._init_F, (len(X), 1))
        for c in range(self.n_classes):
            for tree in self.trees[c]:
                F[:, c] += self.learning_rate * tree.predict(X)
        return self._softmax(F)

    def score(self, X, y):
        preds = self.predict(X)
        if self.task == 'classification':
            return np.mean(preds == y)
        return 1 - np.sum((y - preds) ** 2) / np.sum((y - np.mean(y)) ** 2)
