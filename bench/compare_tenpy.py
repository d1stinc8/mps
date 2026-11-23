"""Item 26: our MPS vs TeNPy on identical problems, each checked against an exact reference.

usage: uv run python bench/compare_tenpy.py            (writes compare_tenpy.csv)

Cases
  quench_exact  N=14 static quench from |0..0>, chi=64 (no truncation: Trotter error only)
  quench_trunc  same, chi=8 (truncation active)
  z2_sweep      N=11 adiabatic Z2 sweep, TeNPy follows H(t) with one engine per step
  long_range    N=12, R=3 (ours only: TeNPy's TEBD needs nearest-neighbor terms)
Reference: scipy expm_multiply (Trotter-free) for static H; midpoint-frozen exact steps for the sweep.
"""

import csv
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "bench"))
import mps  # noqa: E402
import tenpy_rydberg as tr  # noqa: E402
from mps import reference  # noqa: E402

OM, DE, V = 1.0, 0.0, 10.0  # quench parameters (units of Omega)
D0 = 3.0  # sweep: Delta from -D0 to +D0


def infid(a, b):
    """1 - |<a|b>|^2 for (possibly unnormalized) vectors."""
    return 1.0 - abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real)


def dense_obs(v, N):
    """(<n_i>, entropy per bond in bits) from a dense vector."""
    p = np.abs(v) ** 2
    p = p / p.sum()
    bits = (np.arange(len(v))[:, None] >> (N - 1 - np.arange(N))) & 1
    return p @ bits, np.array([reference.entropy(v, b) for b in range(N - 1)])


def run_ours(N, R, T, dt, chi, omega, delta):
    psi = mps.MPS(N)
    t0 = time.perf_counter()
    mps.evolve_rydberg(
        psi,
        V_nn=V,
        R=R,
        omega=omega,
        delta=delta,
        t0=0.0,
        T=T,
        steps=int(round(T / dt)),
        chi_max=chi,
    )
    return psi, time.perf_counter() - t0


def static_case(name, N, R, T, dt, chi, with_tenpy=True):
    ex = reference.evolve_from_zero_krylov(
        reference.rydberg_hamiltonian_sparse(N, V, R, OM, DE), T
    )
    n_ex, S_ex = dense_obs(ex, N)
    ours, sec_o = run_ours(N, R, T, dt, chi, lambda i, t: OM, lambda i, t: DE)
    vo = ours.to_statevector()
    row = dict(
        case=name,
        N=N,
        R=R,
        T=T,
        dt=dt,
        chi=chi,
        infid_ours=infid(ex, vo),
        dn_ours_exact=np.max(np.abs(ours.densities() - n_ex)),
        dS_ours_exact=np.max(np.abs(np.array(ours.entropies()) - S_ex)),
        est_trunc_ours=1 - ours.fidelity,
        bond_ours=max(ours.bond_dims()),
        sec_ours=sec_o,
    )
    if with_tenpy:
        psi, eng, sec_t = tr.evolve_static(N, OM, DE, V, T, dt, chi)
        vt = tr.to_statevector(psi)
        row.update(
            infid_tenpy=infid(ex, vt),
            infid_ours_vs_tenpy=infid(vo, vt),
            dn_tenpy_exact=np.max(np.abs(tr.densities(psi) - n_ex)),
            dS_tenpy_exact=np.max(np.abs(tr.entropies_bits(psi) - S_ex)),
            dn_ours_vs_tenpy=np.max(np.abs(ours.densities() - tr.densities(psi))),
            dS_ours_vs_tenpy=np.max(
                np.abs(np.array(ours.entropies()) - tr.entropies_bits(psi))
            ),
            est_trunc_tenpy=float(eng.trunc_err.eps),
            bond_tenpy=tr.max_bond(psi),
            sec_tenpy=sec_t,
        )
    return row


def sweep_case(N=11, T=32.0, dt=0.05, chi=64):
    def omega(t):
        tr_ = 0.1 * T
        s = (
            np.sin(0.5 * np.pi * t / tr_)
            if t < tr_
            else (np.sin(0.5 * np.pi * (T - t) / tr_) if t > T - tr_ else 1.0)
        )
        return OM * s * s

    def delta(t):
        return -D0 + 2 * D0 * t / T

    ex = reference.evolve_sweep_midpoint(N, V, 1, omega, delta, T, 1600)  # dt = 0.02
    ours, sec_o = run_ours(
        N, 1, T, dt, chi, lambda i, t: omega(t), lambda i, t: delta(t)
    )
    t0 = time.perf_counter()
    psi_t = tr.evolve_sweep(N, V, omega, delta, T, dt, chi)
    sec_t = time.perf_counter() - t0
    vo, vt = ours.to_statevector(), tr.to_statevector(psi_t)
    return dict(
        case="z2_sweep",
        N=N,
        R=1,
        T=T,
        dt=dt,
        chi=chi,
        infid_ours=infid(ex, vo),
        infid_tenpy=infid(ex, vt),
        infid_ours_vs_tenpy=infid(vo, vt),
        order_exact=mps.z2_order(dense_obs(ex, N)[0]),
        order_ours=mps.z2_order(ours.densities()),
        order_tenpy=mps.z2_order(tr.densities(psi_t)),
        bond_ours=max(ours.bond_dims()),
        bond_tenpy=tr.max_bond(psi_t),
        sec_ours=sec_o,
        sec_tenpy=sec_t,
    )


def show(row):
    print(
        f"\n== {row['case']}  N={row['N']} R={row['R']} T={row['T']} dt={row['dt']} chi={row['chi']}"
    )
    for k, v in row.items():
        if k in ("case", "N", "R", "T", "dt", "chi"):
            continue
        print(f"  {k:22s} {v:.4g}" if isinstance(v, float) else f"  {k:22s} {v}")


def main():
    rows = [
        static_case("quench_exact", 14, 1, 4.0, 0.02, 64),
        static_case("quench_trunc", 14, 1, 4.0, 0.02, 8),
        sweep_case(),
        static_case("long_range", 12, 3, 3.0, 0.02, 64, with_tenpy=False),
    ]
    for r in rows:
        show(r)
    cols = sorted(
        {k for r in rows for k in r},
        key=lambda k: (k not in ("case", "N", "R", "T", "dt", "chi"), k),
    )
    with open("compare_tenpy.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    print("\nwrote compare_tenpy.csv")


if __name__ == "__main__":
    main()
