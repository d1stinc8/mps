"""Item 27: runtime vs N, runtime vs chi, error vs chi. Ours vs TeNPy, same problem.

usage (from the project root):
  uv run python bench/bench_scaling.py runtime-n     -> bench_runtime_n.csv
  uv run python bench/bench_scaling.py runtime-chi   -> bench_runtime_chi.csv
  uv run python bench/bench_scaling.py error-chi     -> bench_error_chi.csv
  uv run python bench/bench_scaling.py all
Options: --Ns 10 20 40 80 --chis 4 8 16 32 64 --N 30 --T 4 --dt 0.05 --repeat 3 --chi 32

Problem: quench from |0..0>, H = (Omega/2) sum X - ... + V sum n_i n_{i+1}, Omega=1, Delta=0, V=10.
Both codes run the same number of 2nd-order Trotter steps of the same size dt.
Fairness: every BLAS/LAPACK thread pool is pinned to 1 thread (set before numpy loads).
"""

import os

for _k in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ[_k] = "1"

import argparse  # noqa: E402
import csv  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "python"))
sys.path.insert(0, str(ROOT / "bench"))
import mps  # noqa: E402
import tenpy_rydberg as tr  # noqa: E402
from mps import reference  # noqa: E402

OM, DE, V = 1.0, 0.0, 10.0


def run_ours(N, T, dt, chi):
    """Returns (seconds, psi). Time covers evolve_rydberg only."""
    psi = mps.MPS(N)
    t0 = time.perf_counter()
    mps.evolve_rydberg(
        psi,
        V_nn=V,
        R=1,
        omega=lambda i, t: OM,
        delta=lambda i, t: DE,
        t0=0.0,
        T=T,
        steps=int(round(T / dt)),
        chi_max=chi,
    )
    return time.perf_counter() - t0, psi


def run_tenpy(N, T, dt, chi):
    """Returns (seconds, psi, eps). Time covers TEBDEngine.run() only."""
    psi, eng, sec = tr.evolve_static(N, OM, DE, V, T, dt, chi)
    return sec, psi, float(eng.trunc_err.eps)


def best_of(fn, repeat):
    runs = [fn() for _ in range(repeat)]
    return min(runs, key=lambda r: r[0])


def write(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path}")


def runtime_n(a):
    rows = []
    for N in a.Ns:
        so, po = best_of(lambda: run_ours(N, a.T, a.dt, a.chi), a.repeat)
        st, pt, eps = best_of(lambda: run_tenpy(N, a.T, a.dt, a.chi), a.repeat)
        steps = int(round(a.T / a.dt))
        rows.append(
            dict(
                backend="ours",
                N=N,
                chi=a.chi,
                steps=steps,
                seconds=so,
                max_bond=max(po.bond_dims()),
                est_infid=1 - po.fidelity,
            )
        )
        rows.append(
            dict(
                backend="tenpy",
                N=N,
                chi=a.chi,
                steps=steps,
                seconds=st,
                max_bond=tr.max_bond(pt),
                est_infid=eps,
            )
        )
        print(
            f"N={N:4d}  ours {so:8.3f}s  tenpy {st:8.3f}s  (bond {max(po.bond_dims())}/{tr.max_bond(pt)})"
        )
    write("bench_runtime_n.csv", rows)


def runtime_chi(a):
    rows = []
    for chi in a.chis:
        so, po = best_of(lambda: run_ours(a.N, a.T, a.dt, chi), a.repeat)
        st, pt, eps = best_of(lambda: run_tenpy(a.N, a.T, a.dt, chi), a.repeat)
        steps = int(round(a.T / a.dt))
        rows.append(
            dict(
                backend="ours",
                N=a.N,
                chi=chi,
                steps=steps,
                seconds=so,
                max_bond=max(po.bond_dims()),
                est_infid=1 - po.fidelity,
            )
        )
        rows.append(
            dict(
                backend="tenpy",
                N=a.N,
                chi=chi,
                steps=steps,
                seconds=st,
                max_bond=tr.max_bond(pt),
                est_infid=eps,
            )
        )
        print(
            f"chi={chi:4d}  ours {so:8.3f}s  tenpy {st:8.3f}s  (bond {max(po.bond_dims())}/{tr.max_bond(pt)})"
        )
    write("bench_runtime_chi.csv", rows)


def infid(a, b):
    return 1.0 - abs(np.vdot(a, b)) ** 2 / (np.vdot(a, a).real * np.vdot(b, b).real)


def error_chi(a):
    """Two errors per chi: vs the exact (Krylov) state, which includes the Trotter floor, and vs the
    same code run untruncated (chi = 2^(N/2), the largest possible bond), which isolates truncation."""
    N = a.N
    ex = reference.evolve_from_zero_krylov(
        reference.rydberg_hamiltonian_sparse(N, V, 1, OM, DE), a.T
    )
    chi_full = 2 ** (N // 2)
    _, po_full = run_ours(N, a.T, a.dt, chi_full)
    _, pt_full, _ = run_tenpy(N, a.T, a.dt, chi_full)
    vo_full, vt_full = po_full.to_statevector(), tr.to_statevector(pt_full)
    print(
        f"untruncated (chi={chi_full}) vs exact: ours {infid(ex, vo_full):.3e}  tenpy {infid(ex, vt_full):.3e}"
        "   <- Trotter floor"
    )
    rows = []
    for chi in a.chis:
        so, po = run_ours(N, a.T, a.dt, chi)
        st, pt, eps = run_tenpy(N, a.T, a.dt, chi)
        vo, vt = po.to_statevector(), tr.to_statevector(pt)
        rows.append(
            dict(
                backend="ours",
                N=N,
                chi=chi,
                infid=infid(ex, vo),
                infid_trunc=infid(vo_full, vo),
                est_infid=1 - po.fidelity,
                seconds=so,
            )
        )
        rows.append(
            dict(
                backend="tenpy",
                N=N,
                chi=chi,
                infid=infid(ex, vt),
                infid_trunc=infid(vt_full, vt),
                est_infid=eps,
                seconds=st,
            )
        )
        r_o, r_t = rows[-2], rows[-1]
        print(
            f"chi={chi:4d}  ours: vs exact {r_o['infid']:.2e} trunc-only {r_o['infid_trunc']:.2e} est {r_o['est_infid']:.2e}"
            f" | tenpy: vs exact {r_t['infid']:.2e} trunc-only {r_t['infid_trunc']:.2e} est {r_t['est_infid']:.2e}"
        )
    write("bench_error_chi.csv", rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["runtime-n", "runtime-chi", "error-chi", "all"])
    ap.add_argument("--Ns", type=int, nargs="+", default=[10, 20, 40, 80])
    ap.add_argument("--chis", type=int, nargs="+", default=None)
    ap.add_argument(
        "--N", type=int, default=None, help="chain length for runtime-chi / error-chi"
    )
    ap.add_argument("--chi", type=int, default=32, help="bond cap for runtime-n")
    ap.add_argument("--T", type=float, default=None)
    ap.add_argument("--dt", type=float, default=None)
    ap.add_argument("--repeat", type=int, default=3)
    a = ap.parse_args()

    def preset(mode):
        d = dict(**vars(a))
        if mode == "runtime-n":
            d.update(T=a.T or 4.0, dt=a.dt or 0.05)
        elif mode == "runtime-chi":
            d.update(
                N=a.N or 30,
                T=a.T or 10.0,
                dt=a.dt or 0.05,
                chis=a.chis or [4, 8, 16, 32, 48, 64],
            )
        else:
            d.update(
                N=a.N or 14,
                T=a.T or 6.0,
                dt=a.dt or 0.025,
                chis=a.chis or [2, 4, 8, 16, 32, 64],
            )
        return argparse.Namespace(**d)

    modes = ["runtime-n", "runtime-chi", "error-chi"] if a.mode == "all" else [a.mode]
    fns = {"runtime-n": runtime_n, "runtime-chi": runtime_chi, "error-chi": error_chi}
    for m in modes:
        print(f"\n--- {m}")
        fns[m](preset(m))


if __name__ == "__main__":
    main()
