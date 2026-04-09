import numpy as np
import pytest
from numpycnn import *


@pytest.fixture(autouse=True)
def seed():
    np.random.seed(42)


@pytest.fixture
def spiral_data():
    X, y = make_spiral(100, 3, seed=42)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_tr, X_te, y_tr, y_te


class TestDecisionTree:
    def test_classification(self, spiral_data):
        X_tr, X_te, y_tr, y_te = spiral_data
        dt = DecisionTree(max_depth=8).fit(X_tr, y_tr)
        assert dt.score(X_tr, y_tr) > 0.9

    def test_regression(self):
        X = np.random.randn(100, 2)
        y = X[:, 0] + X[:, 1]
        dt = DecisionTree(max_depth=10, task='regression').fit(X, y)
        assert dt.score(X, y) > 0.8

    def test_predict_proba(self, spiral_data):
        X_tr, X_te, y_tr, y_te = spiral_data
        dt = DecisionTree(max_depth=5).fit(X_tr, y_tr)
        proba = dt.predict_proba(X_te)
        assert proba.shape == (len(X_te), 3)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


class TestRandomForest:
    def test_classification(self, spiral_data):
        X_tr, X_te, y_tr, y_te = spiral_data
        rf = RandomForest(n_estimators=10, max_depth=8, random_state=42).fit(X_tr, y_tr)
        assert rf.score(X_te, y_te) > 0.5

    def test_feature_importances(self, spiral_data):
        X_tr, X_te, y_tr, y_te = spiral_data
        rf = RandomForest(n_estimators=10, random_state=42).fit(X_tr, y_tr)
        imp = rf.feature_importances(X_te, y_te)
        assert imp.shape == (2,)
        assert np.isclose(imp.sum(), 1.0, atol=0.1) or imp.sum() == 0


class TestGBT:
    def test_classification(self, spiral_data):
        X_tr, X_te, y_tr, y_te = spiral_data
        gbt = GradientBoostedTrees(n_estimators=30, max_depth=3, random_state=42).fit(X_tr, y_tr)
        assert gbt.score(X_te, y_te) > 0.8

    def test_regression(self):
        X = np.random.randn(200, 3)
        y = X[:, 0] ** 2 + X[:, 1] - X[:, 2]
        gbt = GradientBoostedTrees(n_estimators=50, max_depth=4, task='regression', random_state=42).fit(X, y)
        assert gbt.score(X, y) > 0.8


class TestKNN:
    def test_classification(self, spiral_data):
        X_tr, X_te, y_tr, y_te = spiral_data
        knn = KNeighborsClassifier(n_neighbors=5).fit(X_tr, y_tr)
        assert knn.score(X_te, y_te) > 0.8

    def test_predict_proba(self, spiral_data):
        X_tr, X_te, y_tr, y_te = spiral_data
        knn = KNeighborsClassifier(3).fit(X_tr, y_tr)
        proba = knn.predict_proba(X_te)
        assert proba.shape[1] == 3
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)


class TestLogisticRegression:
    def test_xor_low_accuracy(self):
        X, y = make_xor(500, seed=42)
        lr = LogisticRegression(lr=0.1, max_iter=200).fit(X, y)
        assert lr.score(X, y) < 0.7

    def test_linearly_separable(self):
        X = np.vstack([np.random.randn(50, 2) + [2, 2], np.random.randn(50, 2) + [-2, -2]])
        y = np.array([0]*50 + [1]*50)
        lr = LogisticRegression(lr=0.1, max_iter=500).fit(X, y)
        assert lr.score(X, y) > 0.95


class TestLinearRegression:
    def test_fit(self):
        X = np.random.randn(100, 3)
        y = 2*X[:, 0] - X[:, 1] + 0.5*X[:, 2] + np.random.randn(100)*0.1
        lr = LinearRegression().fit(X, y)
        assert lr.score(X, y) > 0.99

    def test_ridge(self):
        X = np.random.randn(100, 3)
        y = X[:, 0] + np.random.randn(100)*0.1
        lr = LinearRegression(l2_lambda=0.1).fit(X, y)
        assert lr.score(X, y) > 0.9


class TestPCA:
    def test_dimensionality_reduction(self):
        X = np.random.randn(100, 10)
        pca = PCA(n_components=3)
        X_red = pca.fit_transform(X)
        assert X_red.shape == (100, 3)

    def test_inverse_transform(self):
        X = np.random.randn(50, 5)
        pca = PCA(n_components=5)
        X_red = pca.fit_transform(X)
        X_rec = pca.inverse_transform(X_red)
        np.testing.assert_allclose(X, X_rec, atol=1e-10)

    def test_explained_variance(self):
        X = np.random.randn(100, 10)
        pca = PCA(n_components=5).fit(X)
        assert len(pca.explained_variance_ratio) == 5
        assert pca.explained_variance_ratio[0] >= pca.explained_variance_ratio[-1]


class TestZoo:
    def test_lenet5(self):
        m = LeNet5(10)
        m.compile((None, 28, 28, 1), Adam(), 'he')
        out = m.predict(np.random.randn(2, 28, 28, 1))
        assert out.shape == (2, 10)

    def test_simple_cnn(self):
        m = SimpleCNN(5, filters=[16, 32])
        m.compile((None, 28, 28, 1), Adam(), 'he')
        assert m.predict(np.random.randn(2, 28, 28, 1)).shape == (2, 5)

    def test_lstm_classifier(self):
        m = LSTMClassifier(100, 16, 32, 5)
        m.compile((None, 10), Adam(), 'he')
        assert m.predict(np.random.randint(0, 100, (2, 10))).shape == (2, 5)


class TestSyntheticDatasets:
    def test_spiral(self):
        X, y = make_spiral(50, 3)
        assert X.shape == (150, 2)
        assert set(np.unique(y)) == {0, 1, 2}

    def test_xor(self):
        X, y = make_xor(100)
        assert X.shape == (100, 2)
        assert set(np.unique(y)) == {0, 1}

    def test_sine(self):
        X, y = make_sine_regression(50, 30)
        assert X.shape == (50, 30, 1)
        assert y.shape == (50, 1)

    def test_sequence_classification(self):
        X, y = make_sequence_classification(50, 20, 50, 3)
        assert X.shape == (50, 20)
        assert set(np.unique(y)).issubset({0, 1, 2})


class TestUtils:
    def test_count_parameters(self):
        m = SimpleCNN(10)
        m.compile((None, 28, 28, 1), Adam(), 'he')
        p = count_parameters(m)
        assert p['total'] > 0
        assert p['total'] == p['trainable']

    def test_model_size(self):
        m = SimpleCNN(10)
        m.compile((None, 28, 28, 1), Adam(), 'he')
        assert model_size_bytes(m) > 0
        assert 'MB' in model_size_human(m) or 'KB' in model_size_human(m)
