#pragma once
#include "gates.hpp"
#include <Eigen/Dense>
#include <cmath>
#include <vector>

// Dense state-vector simulator: 2^N amplitudes. Reference for validating the
// MPS. Practical up to N ~ 20 (16 MB at N=20). Qubit 0 = most significant bit.
struct ExactSim {
  int N;
  std::vector<cplx> psi;

  explicit ExactSim(int N_) : N(N_), psi(size_t(1) << N_, cplx(0)) {
    psi[0] = 1.0;
  }

  // Bit of the basis index that holds qubit q.
  size_t mask(int q) const { return size_t(1) << (N - 1 - q); }

  // Basis states i and i|m differ only in qubit q. Each such pair is mixed by
  // U.
  void apply_1q(int q, const Mat2 &U) {
    const size_t m = mask(q);
    for (size_t i = 0; i < psi.size(); ++i)
      if (!(i & m)) {
        const cplx a0 = psi[i], a1 = psi[i | m];
        psi[i] = U[0][0] * a0 + U[0][1] * a1;
        psi[i | m] = U[1][0] * a0 + U[1][1] * a1;
      }
  }

  // Any pair q1 != q2 (adjacent or not). Each group of 4 basis states that
  // differ only in qubits q1, q2 is mixed by U. idx[2*s1 + s2] is the state
  // with qubit q1 = s1 and qubit q2 = s2.
  void apply_2q(int q1, int q2, const Mat4 &U) {
    const size_t m1 = mask(q1), m2 = mask(q2);
    for (size_t i = 0; i < psi.size(); ++i)
      if (!(i & m1) && !(i & m2)) {
        const size_t idx[4] = {i, i | m2, i | m1, i | m1 | m2};
        cplx in[4];
        for (int k = 0; k < 4; ++k)
          in[k] = psi[idx[k]];
        for (int a = 0; a < 4; ++a) {
          cplx s = 0;
          for (int b = 0; b < 4; ++b)
            s += U[a][b] * in[b];
          psi[idx[a]] = s;
        }
      }
  }

  double norm2() const {
    double s = 0;
    for (const cplx &a : psi)
      s += std::norm(a);
    return s;
  }

  // |<psi|other>|^2 / (<psi|psi> <other|other>). 1 = same state up to phase.
  double fidelity(const std::vector<cplx> &other) const {
    cplx ov = 0;
    double no = 0;
    for (size_t i = 0; i < psi.size(); ++i) {
      ov += std::conj(psi[i]) * other[i];
      no += std::norm(other[i]);
    }
    return std::norm(ov) / (norm2() * no);
  }

  // Entropy (bits) across the cut between qubit b and b+1.
  // Row-major reshape: row = first b+1 qubits, column = the rest.
  // Same Schmidt-values logic as MPS::entropy, but with no MPS machinery.
  double entropy(int b) const {
    using RM =
        Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
    const Eigen::Index rows = Eigen::Index(1) << (b + 1);
    const Eigen::Index cols = Eigen::Index(1) << (N - b - 1);
    Eigen::MatrixXcd M = Eigen::Map<const RM>(psi.data(), rows, cols);
    Eigen::BDCSVD<Eigen::MatrixXcd> svd(M);
    Eigen::VectorXd p = svd.singularValues().array().square();
    p /= p.sum();
    double S = 0;
    for (int k = 0; k < p.size(); ++k)
      if (p(k) > 1e-15)
        S -= p(k) * std::log2(p(k));
    return S;
  }
};
