"""TeNPy side of the comparison: same Rydberg chain, same Trotter-2 TEBD.

Conventions match the C++ code: site 0 = most significant bit, state '0' = ground,
'1' = Rydberg, n = |1><1|, H = sum (Omega/2) X - Delta n + sum_{d<=R} V/d^6 n_i n_{i+d}.
"""
import time
import warnings

import numpy as np
from tenpy.algorithms import tebd
from tenpy.linalg import np_conserved as npc
from tenpy.models.lattice import Chain
from tenpy.models.model import CouplingMPOModel, NearestNeighborModel
from tenpy.networks.mps import MPS
from tenpy.networks.site import Site

# TeNPy warns about a new MPS argument that is correct for a Chain.
warnings.filterwarnings("ignore", message="unit_cell_width")


def rydberg_site():
    leg = npc.LegCharge.from_trivial(2)
    return Site(leg, ["0", "1"], X=np.array([[0.0, 1.0], [1.0, 0.0]]), n=np.diag([0.0, 1.0]))


class RydbergChain(CouplingMPOModel, NearestNeighborModel):
    """Nearest neighbor only (R=1): TEBD needs a NearestNeighborModel."""

    default_lattice = Chain
    force_default_lattice = True

    def init_sites(self, model_params):
        return rydberg_site()

    def init_terms(self, model_params):
        Om = model_params.get("Omega", 1.0)
        De = model_params.get("Delta", 0.0)
        V = model_params.get("V", 10.0)
        self.add_onsite(Om / 2.0, 0, "X")
        self.add_onsite(-De, 0, "n")
        self.add_coupling(V, 0, "n", 0, "n", 1)


def make_model(N, Omega, Delta, V):
    return RydbergChain({"L": N, "bc_MPS": "finite", "bc_x": "open",
                         "Omega": Omega, "Delta": Delta, "V": V})


def zero_state(model):
    return MPS.from_product_state(model.lat.mps_sites(), ["0"] * model.lat.N_sites, bc="finite")


def tebd_options(dt, steps, chi_max, svd_min=1e-14):
    return {"order": 2, "dt": dt, "N_steps": steps,
            "trunc_params": {"chi_max": chi_max, "svd_min": svd_min}}


def evolve_static(N, Omega, Delta, V, T, dt, chi_max):
    """Static-H quench from |0...0>. Returns (psi, engine, seconds). steps = round(T/dt).
    Only the evolution is timed (model and state construction excluded)."""
    model = make_model(N, Omega, Delta, V)
    psi = zero_state(model)
    steps = int(round(T / dt))
    eng = tebd.TEBDEngine(psi, model, tebd_options(dt, steps, chi_max))
    t0 = time.perf_counter()
    eng.run()
    return psi, eng, time.perf_counter() - t0


def evolve_sweep(N, V, omega_fn, delta_fn, T, dt, chi_max):
    """Time-dependent H(t): piecewise constant at each step's midpoint, one engine per step.
    The engine rebuilds its bond propagators at construction, so this is the simple
    (slow) way to follow a sweep with TeNPy's TEBD."""
    psi = None
    steps = int(round(T / dt))
    h = T / steps
    for k in range(steps):
        tm = (k + 0.5) * h
        model = make_model(N, omega_fn(tm), delta_fn(tm), V)
        if psi is None:
            psi = zero_state(model)
        eng = tebd.TEBDEngine(psi, model, tebd_options(h, 1, chi_max))
        eng.run()
    return psi


def to_statevector(psi):
    """Dense vector, site 0 = most significant bit. Only for small N."""
    theta = psi.get_theta(0, psi.L)
    theta = theta.itranspose(["vL"] + [f"p{i}" for i in range(psi.L)] + ["vR"])
    return theta.to_ndarray().reshape(-1)


def densities(psi):
    return np.real(psi.expectation_value("n"))


def entropies_bits(psi):
    """Entropy across each internal bond, in bits (TeNPy returns natural log)."""
    return np.asarray(psi.entanglement_entropy()) / np.log(2.0)


def max_bond(psi):
    return int(max(psi.chi))
