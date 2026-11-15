# PA07 Code: invariant subspaces, SVD, and PCA

Complete `spectral.py`. Submit that file or the ZIP made by `make submission`
to **PA07 Code (100 points)**. Submit the fixed PDF separately to **PA07 Written**.
Due dates are in Gradescope. Python 3 and NumPy are sufficient; Matplotlib is
used by the supplied experiment driver. This assignment builds on PA06's QR
work but does not require another C++ library or foreign-function interface.

```sh
python3 -m venv .venv
. .venv/bin/activate
python3 -m pip install -r requirements.txt
make check
make experiments
make submission
```

## Mathematical contracts

All inputs are finite real NumPy arrays of positive dimensions; do not modify
them. Floating-point arrays may be noncontiguous. Return real finite arrays
of the stated shapes. No validation of invalid inputs is required. All norms
below are Frobenius norms unless a subscript says otherwise.

1. `shifted_qr_step(A, mu) -> (B, Q)` (20 points). A is square. Factor
   `A-mu*I = Q@R` and return `B=R@Q+mu*I` with orthogonal Q. Any QR sign
   convention is accepted. Singular shifted matrices are allowed: perform
   full square QR without an inverse. This is one step, not a full eigensolver.
2. `subspace_iteration(A, Q0, max_iter=1000, tol=1e-10)` (35 points).
   A is nonzero n-by-n, Q0 is full-column-rank n-by-r, `1 <= r < n`, and
   every image of the iterated subspace has dimension r. `max_iter` is a
   nonnegative integer; `tol >= 0`. First orthonormalize Q0 with thin QR.
   Form `B=Q.T@A@Q` and `residual=norm(A@Q-Q@B, 'fro')`. If
   `residual <= tol*norm(A,'fro')`, return success **before** any next update.
   Otherwise, while the budget permits, replace Q by the Q factor of A@Q
   and check again. Return `(Q, B, updates, residual, converged)`, with
   shapes (n,r), (r,r), integer, float, Boolean. Count multiplications A@Q
   that update the subspace; initial orthogonalization is update zero.
   At budget zero return the initial orthonormalized subspace and its test.
   Never replace the iteration with an eigendecomposition. A zero residual
   certifies invariance, including an invariant *nondominant* starting space.
   Dominant convergence additionally needs a spectral gap and the appendix's
   initial-subspace condition. Basis columns may rotate or change signs.
3. `svd_lstsq(A, b, rcond=1e-12) -> (x, rank)` (20 points).
   A is m-by-n, b has shape (m,), and `0 <= rcond < 1`.
   From thin SVD retain exactly `s_i > rcond*s_1` (strict inequality).
   Return `x=sum(v_i*(u_i.T@b)/s_i)` over retained indices and their count.
   x has shape (n,). All-zero A returns x=0 and rank=0. When nonzero
   singular values are discarded, x solves the truncated-matrix problem;
   it need not minimize the residual for the original A.
4. `pca(X, k) -> (mean, components, scores, variances)` (25 points).
   X is m-by-n with observations in rows, m>=2, and `1<=k<=min(m-1,n)`.
   Center each column; **do not standardize**. Use SVD of centered X.
   Return the column mean (n,), top k right singular vectors as columns
   (n,k), centered scores (m,k), and ALL min(m,n) sample variances
   `s_i**2/(m-1)`, descending. Reconstruct by `scores@components.T+mean`.
   Constant data are allowed. At repeated singular values, any orthonormal
   optimal subspace is accepted, including a tie across the cutoff.

Use `numpy.linalg.qr` in tasks 1-2 and `numpy.linalg.svd` in tasks 3-4.
Basic NumPy arithmetic, norms, slicing, and matrix products are allowed.
Do not call eig/eigh/eigvals, pinv, lstsq, a ready-made PCA routine, or form
normal equations to implement these tasks. The supplied experiments and
independent tests may use those routines for verification. Grading checks
identities, residuals, optimal approximation error, and subspace projectors,
not a particular choice of signs or basis. Library restrictions also receive
instructor source review; the grader is not a general plagiarism detector.

## Required experiments for Written Q6

`make experiments` writes `build/experiments.json` and `build/subspace.png`.
The driver is supplied so the required work stays in the four kernels.

- Iterate the real 3-by-3 matrix with rotation-dilation block [[3,-4],[4,3]]
  and trailing eigenvalue 1, starting from columns (1,0,1) and (0,1,1).
  Measure projector error at updates 0 through 12, then use tolerance
  1e-10 and budget 100. Compare with the predicted rate 1/5 and recover
  the complex Ritz values from the small B. The driver independently checks
  the residual; convergence concerns the plane, not its rotating basis.
- On the supplied four-observation, three-feature sensor example, compare
  centered rank-two PCA reconstruction with the singular-value tail. The
  offsets (10,20,30) must not enter the variance calculation. This synthetic
  example has an exact answer; no downloaded data are required.
- For diag(1,1e-8), b=(1,1), compare truncated least squares at rcond=1e-6
  and 1e-10. Explain the change in solution norm in the written box.
- Run the supplied larger PCA timing once on your available HPC resource
  (compute node/batch job, respecting local rules). If you lack access, use
  a local machine and identify it explicitly. Record shape, dtype, machine,
  NumPy version, BLAS thread setting, and the median of three calls after
  one warm-up. Compilation, generation, and file I/O are outside the timer.
  Start Python with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.
  There is no speed threshold or performance credit; explain one verification
  check alongside your time. Machine timings need not match the instructor.

The smaller deterministic examples are the mathematical checks. Timing alone
does not establish correctness. Keep evidence and acknowledge AI assistance
and other sources in Written Q6.6; you must understand and verify submitted code.
