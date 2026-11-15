"""Public smoke tests; the autograder also uses independent unseen examples."""
from pathlib import Path
import sys
import unittest
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import spectral as s


class Contracts(unittest.TestCase):
    def test_shift(self):
        A = np.array([[2., 1.], [1., 2.]])
        B, Q = s.shifted_qr_step(A, 1.)
        np.testing.assert_allclose(Q.T @ Q, np.eye(2), atol=1e-12)
        np.testing.assert_allclose(B, Q.T @ A @ Q, atol=1e-12)
        np.testing.assert_allclose(np.tril(Q.T @ (A-np.eye(2)), -1), 0, atol=1e-12)

    def test_plane(self):
        A = np.array([[3., -4., 0.], [4., 3., 0.], [0., 0., 1.]])
        Q, B, k, residual, ok = s.subspace_iteration(A, np.array([[1., 0.], [0., 1.], [1., 1.]]))
        self.assertTrue(ok)
        np.testing.assert_allclose(Q @ Q.T, np.diag([1., 1., 0.]), atol=1e-8)
        np.testing.assert_allclose(B, Q.T @ A @ Q, atol=1e-12)
        self.assertLessEqual(np.linalg.norm(A @ Q-Q @ B), 1e-10*np.linalg.norm(A))

    def test_truncation(self):
        x, rank = s.svd_lstsq(np.diag([1., 1e-8]), np.ones(2), 1e-6)
        np.testing.assert_allclose(x, [1., 0.], atol=1e-12)
        self.assertEqual(rank, 1)

    def test_pca(self):
        X = np.array([[12.,20.,31.],[8.,20.,31.],[10.,21.,29.],[10.,19.,29.]])
        mean, V, scores, variance = s.pca(X, 2)
        np.testing.assert_allclose(mean, [10.,20.,30.])
        np.testing.assert_allclose(variance, [8/3,4/3,2/3])
        self.assertAlmostEqual(np.linalg.norm(X-(scores@V.T+mean))**2, 2.)


if __name__ == "__main__": unittest.main()
