"""Reproducible mathematical checks and a separately reported PCA timing."""
import json
import os
from pathlib import Path
import platform
import sys
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import spectral as s


def main():
    A = np.array([[3.,-4.,0.],[4.,3.,0.],[0.,0.,1.]])
    Q0 = np.array([[1.,0.],[0.,1.],[1.,1.]])
    target = np.diag([1.,1.,0.])
    errors = []
    for k in range(13):
        Q, B, updates, residual, ok = s.subspace_iteration(A,Q0,k,0.)
        errors.append(float(np.linalg.norm(Q@Q.T-target)))
    Q, B, k, residual, ok = s.subspace_iteration(A,Q0,100,1e-10)
    actual_residual = float(np.linalg.norm(A@Q-Q@B))
    assert ok and actual_residual <= 1.00001e-10*np.linalg.norm(A)
    assert np.allclose(Q.T@Q,np.eye(2))
    eig = np.linalg.eigvals(B)
    X = np.array([[12.,20.,31.],[8.,20.,31.],[10.,21.,29.],[10.,19.,29.]])
    mean,V,scores,var = s.pca(X,2)
    pca_error = float(np.linalg.norm(X-(scores@V.T+mean))**2)
    assert np.isclose(pca_error, 2.)
    assert np.allclose(var,[8/3,4/3,2/3])
    solutions = []
    for cutoff in (1e-6,1e-10):
        x, rank = s.svd_lstsq(np.diag([1.,1e-8]),np.ones(2),cutoff)
        solutions.append(dict(rcond=cutoff,x=x.tolist(),rank=int(rank),norm=float(np.linalg.norm(x))))
    rng = np.random.default_rng(58907)
    data = rng.normal(size=(4000,100)) @ np.diag(np.linspace(4,1,100)) + 10
    s.pca(data,10)
    times = []
    for _ in range(3):
        start = time.perf_counter()
        mean,V,scores,var = s.pca(data,10)
        times.append(time.perf_counter()-start)
    err = float(np.linalg.norm(data-(scores@V.T+mean))**2)
    tail = float((len(data)-1)*var[10:].sum())
    assert np.isclose(err,tail,rtol=1e-10)
    result = dict(plane=dict(updates=k,residual=actual_residual,reported_residual=residual,
                            ritz_values=[[float(z.real),float(z.imag)] for z in eig],
                            projector_errors=errors,last_ratio=errors[-1]/errors[-2]),
                  pca=dict(mean=[10.,20.,30.],squared_error=pca_error,retained_fraction=6/7),
                  truncated_solutions=solutions,
                  timing=dict(seconds=times,median_seconds=float(np.median(times)),shape=list(data.shape),
                              dtype=str(data.dtype),machine=platform.node(),cpu=platform.processor(),
                              platform=platform.platform(),numpy=np.__version__,
                              threads={name:os.environ.get(name,"unset") for name in
                                       ("OPENBLAS_NUM_THREADS","OMP_NUM_THREADS","MKL_NUM_THREADS")},
                              squared_error=err,tail=tail))
    out = ROOT/"build"
    out.mkdir(exist_ok=True)
    (out/"experiments.json").write_text(json.dumps(result,indent=2)+"\n")
    fig,ax = plt.subplots(figsize=(6,4))
    ax.semilogy(range(13),errors,"o-",label="projector error")
    ax.semilogy(range(13),errors[0]*.2**np.arange(13),"--",label="reference rate (1/5)^k")
    ax.set(xlabel="Subspace updates",ylabel="Frobenius projector distance")
    ax.legend(); fig.tight_layout(); fig.savefig(out/"subspace.png",dpi=160); plt.close(fig)
    print(json.dumps(result,indent=2))


if __name__ == "__main__": main()
