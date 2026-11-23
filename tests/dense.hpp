#pragma once
#include "exact.hpp"
#include "mps.hpp"

// Thin wrappers so older tests keep working; the logic lives in ExactSim.
inline std::vector<cplx> dense_1q(std::vector<cplx> v, int N, int q,
                                  const Mat2 &U) {
  ExactSim e(N);
  e.psi = std::move(v);
  e.apply_1q(q, U);
  return std::move(e.psi);
}
inline std::vector<cplx> dense_2q(std::vector<cplx> v, int N, int q,
                                  const Mat4 &U) {
  ExactSim e(N);
  e.psi = std::move(v);
  e.apply_2q(q, q + 1, U);
  return std::move(e.psi);
}
