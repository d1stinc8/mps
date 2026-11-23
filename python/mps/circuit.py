import numpy as np

from . import gates
from ._mps import MPS


class Circuit:
    """Gate list; run() applies it to an MPS. Qubit 0 is the most significant bit."""

    def __init__(self, n):
        if n < 1:
            raise ValueError("n must be >= 1")
        self.n = n
        self.ops = []  # ("1q", q, U) or ("2q", q1, q2, U)

    def _check(self, *qs):
        for q in qs:
            if not 0 <= q < self.n:
                raise IndexError(f"qubit {q} out of range for n={self.n}")

    def gate1(self, q, U):
        self._check(q)
        self.ops.append(("1q", q, np.asarray(U, dtype=complex)))
        return self

    def gate2(self, q1, q2, U):
        self._check(q1, q2)
        if q1 == q2:
            raise ValueError("q1 and q2 must differ")
        self.ops.append(("2q", q1, q2, np.asarray(U, dtype=complex)))
        return self

    def h(self, q):
        return self.gate1(q, gates.H)

    def x(self, q):
        return self.gate1(q, gates.X)

    def z(self, q):
        return self.gate1(q, gates.Z)

    def rx(self, q, t):
        return self.gate1(q, gates.Rx(t))

    def rz(self, q, t):
        return self.gate1(q, gates.Rz(t))

    def cz(self, a, b):
        return self.gate2(a, b, gates.CZ)

    def cnot(self, ctrl, tgt):
        return self.gate2(ctrl, tgt, gates.CNOT)

    def swap(self, a, b):
        return self.gate2(a, b, gates.SWAP)

    def cphase(self, a, b, phi):
        return self.gate2(a, b, gates.CPhase(phi))

    def run(self, chi_max=None, cutoff=1e-14, seed=None):
        """Apply all gates to |0...0>. chi_max=None: no bond cap."""
        psi = MPS(self.n, seed)
        for op in self.ops:
            if op[0] == "1q":
                psi.apply_1q(op[1], op[2])
            else:
                psi.apply_2q(op[1], op[2], op[3], chi_max=chi_max, cutoff=cutoff)
        return psi
