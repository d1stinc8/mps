"""Gate matrices. 2q index = 2*s(first qubit) + s(second qubit)."""
import numpy as np

H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
X = np.array([[0, 1], [1, 0]], dtype=complex)
Z = np.array([[1, 0], [0, -1]], dtype=complex)
CZ = np.diag([1, 1, 1, -1]).astype(complex)
CNOT = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
SWAP = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]], dtype=complex)


def Rx(t):
    c, s = np.cos(t / 2), -1j * np.sin(t / 2)
    return np.array([[c, s], [s, c]], dtype=complex)


def Rz(t):
    return np.diag([np.exp(-0.5j * t), np.exp(0.5j * t)])


def CPhase(phi):
    return np.diag([1, 1, 1, np.exp(1j * phi)])
