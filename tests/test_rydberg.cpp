#include "exact.hpp"
#include "rydberg.hpp"
#include <gtest/gtest.h>

static RydbergChain make_chain(int N, double omega, double delta, double V) {
  RydbergChain H;
  H.N = N;
  H.V.assign(N - 1, V);
  H.Omega = [=](int, double) { return omega; };
  H.Delta = [=](int, double) { return delta; };
  return H;
}

static const double PI = 3.14159265358979323846;

// Dense H(t) from the definition. Site i = bit N-1-i, like ExactSim.
static Eigen::MatrixXcd build_H(const RydbergChain &H, double t) {
  const int N = H.N;
  const size_t D = size_t(1) << N;
  Eigen::MatrixXcd M = Eigen::MatrixXcd::Zero(D, D);
  for (size_t b = 0; b < D; ++b) {
    double diag = 0;
    for (int i = 0; i < N; ++i) {
      const size_t mi = size_t(1) << (N - 1 - i);
      const bool ni = (b & mi) != 0;
      if (ni)
        diag -= H.Delta(i, t);
      if (i + 1 < N && ni && (b & (mi >> 1)))
        diag += H.V[i];
      M(b ^ mi, b) += 0.5 * H.Omega(i, t); // (Omega/2) X_i flips bit i
    }
    for (size_t k = 0; k < H.Vlong.size(); ++k) { // long-range terms
      const int d = int(k) + 2;
      for (int i = 0; i + d < N; ++i) {
        const size_t mi = size_t(1) << (N - 1 - i), mj = size_t(1)
                                                         << (N - 1 - (i + d));
        if ((b & mi) && (b & mj))
          diag += H.Vlong[k][i];
      }
    }
    M(b, b) += diag;
  }
  return M;
}

// exp(-i T H) |0..0>, H(t) frozen at each step's midpoint. Independent of the
// gate formulas: only the Hamiltonian definition and an eigendecomposition.
static std::vector<cplx> exact_evolve(const RydbergChain &H, double T,
                                      int steps = 1) {
  const size_t D = size_t(1) << H.N;
  Eigen::VectorXcd psi = Eigen::VectorXcd::Zero(D);
  psi(0) = 1.0;
  const double dt = T / steps;
  for (int k = 0; k < steps; ++k) {
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXcd> es(
        build_H(H, (k + 0.5) * dt));
    Eigen::VectorXcd c = es.eigenvectors().adjoint() * psi;
    for (size_t j = 0; j < D; ++j)
      c(j) *= std::polar(1.0, -es.eigenvalues()(j) * dt);
    psi = es.eigenvectors() * c;
  }
  return std::vector<cplx>(psi.data(), psi.data() + D);
}

TEST(Rydberg, DriveGateLimits) {
  const double dt = 0.37;
  { // Delta = 0: Rx(omega dt)
    Mat2 A = drive_gate(1.3, 0.0, dt), B = gates::Rx(1.3 * dt);
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j)
        EXPECT_NEAR(std::abs(A[i][j] - B[i][j]), 0.0, 1e-12);
  }
  { // Omega = 0: diag(1, e^{+i delta dt}); only |1> sees the detuning
    Mat2 C = drive_gate(0.0, 0.8, dt);
    EXPECT_NEAR(std::abs(C[0][0] - 1.0), 0.0, 1e-12);
    EXPECT_NEAR(std::abs(C[1][1] - std::polar(1.0, 0.8 * dt)), 0.0, 1e-12);
    EXPECT_NEAR(std::abs(C[0][1]), 0.0, 1e-12);
  }
  { // unitary
    Mat2 U = drive_gate(1.1, -0.7, 0.2);
    for (int i = 0; i < 2; ++i)
      for (int j = 0; j < 2; ++j) {
        cplx s = 0;
        for (int k = 0; k < 2; ++k)
          s += std::conj(U[k][i]) * U[k][j];
        EXPECT_NEAR(std::abs(s - (i == j ? 1.0 : 0.0)), 0.0, 1e-12);
      }
  }
}

// Second order: halving dt cuts the state error 4x, so the infidelity 16x.
TEST(Rydberg, TrotterIsSecondOrder) {
  const int N = 6;
  const double T = 1.0;
  auto H = make_chain(N, 2.0, 1.0, 3.0);
  ExactSim e(N);
  e.psi = exact_evolve(H, T);
  auto infid = [&](int steps) {
    MPS psi(N);
    evolve(psi, H, 0.0, T, steps);
    return 1.0 - e.fidelity(psi.to_statevector());
  };
  const double e10 = infid(10), e20 = infid(20), e200 = infid(200);
  EXPECT_GT(e10, 1e-8);      // not trivially exact
  EXPECT_GT(e10 / e20, 8.0); // ~16 expected
  EXPECT_LT(e200, 1e-6);
}

// Same Trotter circuit on MPS and on the dense vector, with site- and
// time-dependent drive: isolates the MPS from Trotter error.
TEST(Rydberg, MPSMatchesDenseSameCircuit) {
  const int N = 8, steps = 30;
  const double dt = 0.05;
  RydbergChain H;
  H.N = N;
  H.V.assign(N - 1, 2.5);
  H.Omega = [](int i, double t) { return 1.0 + 0.2 * i + 0.5 * std::sin(t); };
  H.Delta = [](int i, double t) { return -0.4 + 0.1 * i * t; };

  ExactSim e(N);
  for (int k = 0; k < steps; ++k)
    strang_step(
        H, k * dt, dt, [&](int q, const Mat2 &U) { e.apply_1q(q, U); },
        [&](int q, const Mat4 &U) { e.apply_2q(q, q + 1, U); });

  MPS psi(N);
  evolve(psi, H, 0.0, steps * dt, steps);
  EXPECT_NEAR(e.fidelity(psi.to_statevector()), 1.0, 1e-10);
  EXPECT_NEAR(psi.fidelity, 1.0, 1e-10);
  EXPECT_NEAR(psi.norm(), 1.0, 1e-10);
  auto S = psi.entropies();
  for (int b = 0; b + 1 < N; ++b)
    EXPECT_NEAR(S[b], e.entropy(b), 1e-8);
}

// Time-dependent sweep vs an independent exact reference.
// Evaluating the drive at t instead of the midpoint would make this first order
// (ratio ~4) and fail the check below.
TEST(Rydberg, TimeDependentSweepIsSecondOrder) {
  const int N = 5;
  const double T = 2.0;
  RydbergChain H;
  H.N = N;
  H.V.assign(N - 1, 4.0);
  H.Omega = [=](int i, double t) {
    return 2.0 * (1.0 + 0.1 * i) * std::sin(PI * t / T);
  };
  H.Delta = [=](int, double t) { return -3.0 + 6.0 * t / T; };

  ExactSim ref(N);
  ref.psi = exact_evolve(H, T, 800);
  EXPECT_NEAR(ref.fidelity(exact_evolve(H, T, 400)), 1.0,
              1e-9); // reference converged

  auto infid = [&](int steps) {
    MPS psi(N);
    evolve(psi, H, 0.0, T, steps);
    return 1.0 - ref.fidelity(psi.to_statevector());
  };
  const double e20 = infid(20), e40 = infid(40), e160 = infid(160);
  EXPECT_GT(e20, 1e-6);         // not trivially exact
  EXPECT_GT(e20 / e40, 8.0);    // ~16 expected for second order
  EXPECT_GT(e40 / e160, 100.0); // ~256 expected
}
// Isolates the swap network and the long-range indices: same Trotter circuit,
// two backends.
TEST(Rydberg, LongRangeMPSMatchesDenseSameCircuit) {
  const int N = 7, steps = 20;
  const double dt = 0.05;
  auto H = make_vdw_chain(
      N, 6.0, 3,
      [](int i, double t) { return 1.0 + 0.1 * i + 0.3 * std::sin(t); },
      [](int, double t) { return -1.0 + 0.5 * t; });
  ExactSim e(N);
  for (int k = 0; k < steps; ++k)
    strang_step_lr(
        H, k * dt, dt, [&](int q, const Mat2 &U) { e.apply_1q(q, U); },
        [&](int a, int b, const Mat4 &U) { e.apply_2q(a, b, U); });
  MPS psi(N);
  evolve(psi, H, 0.0, steps * dt, steps);
  EXPECT_NEAR(e.fidelity(psi.to_statevector()), 1.0, 1e-10);
  EXPECT_NEAR(psi.fidelity, 1.0, 1e-10);
  EXPECT_NEAR(psi.norm(), 1.0, 1e-10);
}

// Long-range H built from its definition: Trotter must converge to it at second
// order.
TEST(Rydberg, LongRangeTrotterConvergesToExact) {
  const int N = 6;
  const double T = 1.0;
  auto H = make_vdw_chain(
      N, 8.0, 3, [](int, double) { return 2.0; },
      [](int, double) { return 1.0; });
  ExactSim e(N);
  e.psi = exact_evolve(H, T);
  auto infid = [&](int steps) {
    MPS psi(N);
    evolve(psi, H, 0.0, T, steps);
    return 1.0 - e.fidelity(psi.to_statevector());
  };
  const double e10 = infid(10), e20 = infid(20), e200 = infid(200);
  EXPECT_GT(e10, 1e-8);
  EXPECT_GT(e10 / e20, 8.0);
  EXPECT_LT(e200, 1e-6);
}

// Error of dropping couplings beyond R, vs the full 1/r^6 chain. Shrinks fast
// with R.
TEST(Rydberg, CutoffErrorShrinksWithR) {
  const int N = 7;
  const double T = 2.0;
  auto build = [&](int R) {
    return make_vdw_chain(
        N, 10.0, R, [](int, double) { return 1.0; },
        [](int, double) { return 0.5; });
  };
  ExactSim full(N);
  full.psi = exact_evolve(build(N - 1), T);
  double err[3];
  for (int R = 1; R <= 3; ++R)
    err[R - 1] = 1.0 - full.fidelity(exact_evolve(build(R), T));
  std::cout << "cutoff infidelity R=1,2,3: " << err[0] << " " << err[1] << " "
            << err[2] << "\n";
  EXPECT_GT(err[0], err[1]);
  EXPECT_GT(err[1], err[2]);
}
