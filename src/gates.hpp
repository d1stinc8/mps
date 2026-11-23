#pragma once
#include <array>
#include <cmath>
#include <complex>

using cplx = std::complex<double>;
using Mat2 = std::array<std::array<cplx, 2>, 2>; // U[out][in]
using Mat4 = std::array<std::array<cplx, 4>, 4>; // index = 2*s_first + s_second

namespace gates {

inline Mat2 H() {
  const double h = 1.0 / std::sqrt(2.0);
  return {{{h, h}, {h, -h}}};
}
inline Mat2 X() { return {{{0, 1}, {1, 0}}}; }
inline Mat2 Z() { return {{{1, 0}, {0, -1}}}; }

// Rx(t) = exp(-i t X / 2): rotation about x. Rabi drive acts like this.
inline Mat2 Rx(double t) {
  const cplx c = std::cos(t / 2), s = cplx(0, -1) * std::sin(t / 2);
  return {{{c, s}, {s, c}}};
}
// Rz(t) = exp(-i t Z / 2): phase rotation. Detuning acts like this.
inline Mat2 Rz(double t) {
  return {{{std::polar(1.0, -t / 2), 0}, {0, std::polar(1.0, t / 2)}}};
}
// CZ: phase -1 only on |11>. Rydberg blockade gate is this up to 1q rotations.
inline Mat4 CZ() {
  Mat4 m{};
  m[0][0] = m[1][1] = m[2][2] = 1;
  m[3][3] = -1;
  return m;
}

inline Mat4 CNOT() {
  Mat4 m{};
  m[0][0] = m[1][1] = 1;
  m[2][3] = m[3][2] = 1;
  return m;
}

inline Mat4 SWAP() {
  Mat4 m{};
  m[0][0] = m[3][3] = 1;
  m[1][2] = m[2][1] = 1;
  return m;
}

// U'[(a,b)][(c,d)] = U[(b,a)][(d,c)]. Used when the first qubit is the right
// one.
inline Mat4 swap_order(const Mat4 &U) {
  Mat4 m{};
  for (int a = 0; a < 2; ++a) {
    for (int b = 0; b < 2; ++b) {
      for (int c = 0; c < 2; ++c) {
        for (int d = 0; d < 2; ++d) {
          m[2 * a + b][2 * c + d] = U[2 * b + a][2 * d + c];
        }
      }
    }
  }

  return m;
}

// Controlled phase: diag(1, 1, 1, e^{i phi}). Symmetric in its two qubits.
inline Mat4 CPhase(double phi) {
  Mat4 m{};
  m[0][0] = m[1][1] = m[2][2] = 1;
  m[3][3] = std::polar(1.0, phi);
  return m;
}

} // namespace gates
