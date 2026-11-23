"""Dense/sparse numpy references for validation. Qubit 0 = most significant bit."""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import expm_multiply


def dense_state(n, ops):
    """Apply Circuit.ops to |0...0> on a (2,)*n tensor. Axis q = qubit q."""
    psi = np.zeros((2,) * n, dtype=complex)
    psi[(0,) * n] = 1.0
    for op in ops:
        if op[0] == "1q":
            _, q, U = op
            psi = np.moveaxis(np.tensordot(U, psi, axes=([1], [q])), 0, q)
        else:
            _, q1, q2, U = op
            res = np.tensordot(U.reshape(2, 2, 2, 2), psi, axes=([2, 3], [q1, q2]))
            psi = np.moveaxis(res, [0, 1], [q1, q2])
    return psi.reshape(-1)


def entropy(state, b):
    """Entropy (bits) across the cut between qubit b and b+1."""
    n = int(np.log2(state.size))
    s = np.linalg.svd(state.reshape(2 ** (b + 1), 2 ** (n - b - 1)), compute_uv=False)
    p = s**2 / np.sum(s**2)
    p = p[p > 1e-15]
    return float(-(p * np.log2(p)).sum())


def rydberg_hamiltonian(n, V_nn, R, omega, delta):
    """Static H = sum (omega/2) X_i - delta n_i + sum_{d<=R} V_nn/d^6 n_i n_{i+d}. Dense."""
    return rydberg_hamiltonian_sparse(n, V_nn, R, omega, delta).toarray()


def rydberg_hamiltonian_sparse(n, V_nn, R, omega, delta):
    """Same H as a scipy sparse matrix: usable up to N ~ 20."""
    D = 1 << n
    idx = np.arange(D)
    bits = (
        idx[:, None] >> (n - 1 - np.arange(n))
    ) & 1  # bits[b, i] = n_i in basis state b
    diag = -delta * bits.sum(axis=1).astype(float)
    for d in range(1, R + 1):
        diag += (V_nn / d**6) * (bits[:, : n - d] * bits[:, d:]).sum(axis=1)
    H = sp.diags(diag).astype(complex).tocsr()
    for i in range(n):
        flip = idx ^ (1 << (n - 1 - i))  # (omega/2) X_i flips bit i
        H = H + sp.csr_matrix(
            (np.full(D, 0.5 * omega, dtype=complex), (flip, idx)), shape=(D, D)
        )
    return H


def evolve_from_zero(H, T):
    """exp(-i H T)|0...0> by diagonalization. Dense H only (N <= 12)."""
    w, v = np.linalg.eigh(H)
    return v @ (np.exp(-1j * w * T) * v[0].conj())


def evolve_from_zero_krylov(H_sparse, T):
    """exp(-i H T)|0...0> with scipy's expm_multiply: Trotter-free, N up to ~18."""
    psi0 = np.zeros(H_sparse.shape[0], dtype=complex)
    psi0[0] = 1.0
    return expm_multiply(-1j * T * H_sparse, psi0)


def evolve_sweep_midpoint(n, V_nn, R, omega_fn, delta_fn, T, steps):
    """exp(-i H(t) T)|0..0> for a time-dependent sweep: H frozen at each step's midpoint,
    each step exact (Krylov). omega_fn(t), delta_fn(t). Error is O(dt^2): use many steps."""
    psi = np.zeros(1 << n, dtype=complex)
    psi[0] = 1.0
    dt = T / steps
    for k in range(steps):
        tm = (k + 0.5) * dt
        H = rydberg_hamiltonian_sparse(n, V_nn, R, omega_fn(tm), delta_fn(tm))
        psi = expm_multiply(-1j * dt * H, psi)
    return psi
