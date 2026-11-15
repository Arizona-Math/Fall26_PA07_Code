"""PA07 Code: implement these four mathematical kernels. See the starter README.

Inputs are finite real arrays satisfying the stated assumptions. Do not mutate
them. NumPy QR/SVD are allowed as specified; no general validation is required.
"""
import numpy as np


def shifted_qr_step(A, mu):
    """Return (B, Q): A-mu*I=Q*R and B=R*Q+mu*I, using full square QR."""
    raise NotImplementedError("one shifted QR step")


def subspace_iteration(A, Q0, max_iter=1000, tol=1e-10):
    """Return (Q, B, updates, residual, converged).

    Orthonormalize Q0, then repeatedly replace Q by thin_qr(A@Q)[0].
    B=Q.T@A@Q; residual=norm(A@Q-Q@B, 'fro'). Check at update 0 and
    after every update; stop at the first residual <= tol*norm(A, 'fro'),
    or at max_iter. The residual certifies invariance, not dominance.
    """
    raise NotImplementedError("modified power method")


def svd_lstsq(A, b, rcond=1e-12):
    """Return (x, rank) using only singular values s > rcond*s[0].

    x is the minimum-norm least-squares solution for the truncated matrix.
    All-zero A gives zero x and rank 0. Use a thin library SVD, not pinv/lstsq.
    """
    raise NotImplementedError("truncated SVD least squares")


def pca(X, k):
    """Return (mean, components, scores, variances) for centered-only PCA.

    Rows are observations. components has shape (n_features,k); scores is
    (X-mean)@components. variances contains ALL min(X.shape) values
    s**2/(n_observations-1), in descending order. Do not standardize.
    """
    raise NotImplementedError("PCA through the SVD")
