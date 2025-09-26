import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

def optimal_permutation(X: torch.Tensor, recon_X: torch.Tensor):
    n = X.shape[0]
    m = recon_X.shape[0]

    # 1) Flatten spatial dims, compute pairwise squared‐L2 distances
    flat_X     = X.view(n, -1)                 # (n, D)
    flat_recon = recon_X.view(m, -1)           # (m, D)
    # cost_matrix[i, j] = || X[i] - recon_X[j] ||^2
    cost_matrix = (
        torch.cdist(flat_X, flat_recon, p=2.0)
             .pow(2)
             .cpu()
             .numpy()
    )                                           # shape: (n, m)

    # 2) Hungarian algorithm on rectangular cost matrix
    row_ind, col_ind = linear_sum_assignment(cost_matrix)
    #    row_ind[k] in [0..n), col_ind[k] in [0..m), with len(row_ind)==n

    # 3) Build perm so that perm[i] = index in recon_X matched to X[i]
    perm = torch.empty(n, dtype=torch.long)
    # row_ind[k] = i  matched to  col_ind[k] = j
    perm[row_ind] = torch.from_numpy(col_ind)
    return perm


def orth_rowspan_residuals(A, Y, p):
    A = np.asarray(A)
    Y = np.asarray(Y)
    if A.ndim != 2 or Y.ndim != 2:
        raise ValueError("Both inputs must be 2-D arrays.")
    if A.shape[1] != Y.shape[1]:
        raise ValueError("Number of columns must match (same ambient dimension).")

    # SVD of A; row-space(A) == span of first r right singular vectors
    U, s, Vt = np.linalg.svd(A, full_matrices=False)

    # Numerical rank, same rule as numpy.linalg.matrix_rank
    eps = np.finfo(s.dtype).eps
    tol_rank = s.max() * max(A.shape) * eps if s.size else 0.0
    r = int(np.sum(s > tol_rank))

    # Orthonormal basis for row space (columns of V corresponding to nonzero singular values)
    Vr = Vt[:r, :].T                     # shape (p, r)
    H = Vr.conj().T                      # handles real/complex

    # Orthogonal projection of Y onto row-space(A)
    Y_proj = Y @ Vr @ H                  # shape (m, p)

    # Residuals
    return np.linalg.norm(Y - Y_proj, axis=1) / (p**0.5)