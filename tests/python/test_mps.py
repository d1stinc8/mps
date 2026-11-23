import numpy as np
import pytest

import mps
from mps import Circuit, gates, reference


def random_circuit(n, steps, seed):
    rng = np.random.default_rng(seed)
    c = Circuit(n)
    for _ in range(steps):
        k = rng.integers(4)
        if k == 0:
            c.rx(int(rng.integers(n)), rng.uniform(0, 2 * np.pi))
        elif k == 1:
            c.rz(int(rng.integers(n)), rng.uniform(0, 2 * np.pi))
        else:
            a, b = (int(x) for x in rng.choice(n, size=2, replace=False))  # any pair
            if k == 2:
                c.cnot(a, b)
            else:
                c.cphase(a, b, rng.uniform(0, 2 * np.pi))
    return c


def test_bell():
    psi = Circuit(2).h(0).cnot(0, 1).run()
    h = 1 / np.sqrt(2)
    assert np.allclose(psi.to_statevector(), [h, 0, 0, h], atol=1e-12)
    assert psi.bond_dims() == [2]
    assert psi.entropy(0) == pytest.approx(1.0, abs=1e-10)


@pytest.mark.parametrize("seed", range(5))
def test_random_circuit_matches_numpy(seed):
    n = 7
    c = random_circuit(n, 60, seed)
    psi = c.run()
    ref = reference.dense_state(n, c.ops)
    assert np.max(np.abs(psi.to_statevector() - ref)) < 1e-10
    assert psi.fidelity == pytest.approx(1.0, abs=1e-10)


def test_entropy_matches_numpy_svd():
    n = 8
    c = random_circuit(n, 80, 11)
    psi = c.run()
    ref = reference.dense_state(n, c.ops)
    for b in range(n - 1):
        assert psi.entropy(b) == pytest.approx(reference.entropy(ref, b), abs=1e-8)


def test_ghz_samples():
    n = 6
    c = Circuit(n).h(0)
    for i in range(n - 1):
        c.cnot(i, i + 1)
    s = c.run(seed=1).sample(2000)
    assert s.shape == (2000, n) and s.dtype == np.uint8
    assert np.all(s == s[:, :1])  # all bits equal in every shot
    assert 800 < int(s[:, 0].sum()) < 1200  # both outcomes, about 50/50


def test_sampling_matches_born_rule():
    n = 5
    c = random_circuit(n, 40, 2)
    ref = reference.dense_state(n, c.ops)
    s = c.run(seed=3).sample(50000).astype(np.int64)
    idx = s @ (1 << np.arange(n - 1, -1, -1))  # qubit 0 = most significant bit
    freq = np.bincount(idx, minlength=2**n) / 50000
    assert np.max(np.abs(freq - np.abs(ref) ** 2)) < 0.01


def test_seed_is_reproducible():
    c = random_circuit(6, 40, 4)
    a = c.run(seed=9).sample(20)
    b = c.run(seed=9).sample(20)
    assert np.array_equal(a, b)


def test_chi_cap():
    c = random_circuit(8, 100, 3)
    psi = c.run(chi_max=4)
    assert max(psi.bond_dims()) <= 4
    assert 0 < psi.fidelity < 1
    assert psi.norm() == pytest.approx(1.0, abs=1e-10)


def test_evolve_rydberg_vs_exact():
    n, T = 6, 1.0
    om, de, V = 2.0, 1.0, 8.0
    psi = mps.MPS(n)
    mps.evolve_rydberg(
        psi,
        V_nn=V,
        R=3,
        omega=lambda i, t: om,
        delta=lambda i, t: de,
        t0=0.0,
        T=T,
        steps=200,
    )
    ref = reference.evolve_from_zero(reference.rydberg_hamiltonian(n, V, 3, om, de), T)
    assert abs(np.vdot(ref, psi.to_statevector())) ** 2 > 1 - 1e-6


def test_z2_order():
    assert mps.z2_order([1, 0, 1, 0, 1]) == pytest.approx(1.0)
    assert mps.z2_order([0, 1, 0, 1, 0]) == pytest.approx(-1.0)


def test_bad_inputs():
    psi = mps.MPS(3)
    with pytest.raises(IndexError):
        psi.apply_1q(3, gates.H)
    with pytest.raises(ValueError):
        psi.apply_1q(0, np.eye(3))
    with pytest.raises(ValueError):
        psi.apply_2q(1, 1, gates.CZ)
    with pytest.raises(ValueError):
        psi.apply_2q(0, 1, gates.CZ, chi_max=0)
