import unittest
from cefi.core import cluster_enhanced_feature_importance
from sklearn.datasets import make_classification

class TestCEFI(unittest.TestCase):
    def test_importance(self):
        X, _ = make_classification(n_samples=1000, n_features=10, n_informative=5, random_state=42)
        importance = cluster_enhanced_feature_importance(X, n_clusters=3)
        self.assertEqual(len(importance), 10)

if __name__ == '__main__':
    unittest.main()