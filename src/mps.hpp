#pragma once
#include "gates.hpp"
#include <Eigen/Dense>
#include <algorithm>
#include <cassert>
#include <complex>
#include <random>
#include <utility>
#include <vector>

// Row-major matrix. Our flat layout (l*2+s)*Dr + r is already a row-major
// matrix
using RowMat =
    Eigen::Matrix<cplx, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;

// One MPS site tensor A[l][s][r]:
//   l = left bond index  (size Dl)
//   s = physical index   (size 2: |0>, |1>)
//   r = right bond index (size Dr)
// Stored flat (row-major): index = (l*2 + s)*Dr + r
struct Tensor {
  int Dl = 1, Dr = 1;
  std::vector<cplx> data;

  Tensor(int Dl_, int Dr_) : Dl(Dl_), Dr(Dr_), data(Dl_ * 2 * Dr_, 0.0) {}

  cplx &at(int l, int s, int r) { return data[(l * 2 + s) * Dr + r]; }
  cplx at(int l, int s, int r) const { return data[(l * 2 + s) * Dr + r]; }

  // View as matrix, grouping (l, s) as rows, r as columns: (Dl * 2) x Dr.
  Eigen::MatrixXcd left_mat() const {
    return Eigen::Map<const RowMat>(data.data(), Dl * 2, Dr);
  }
  // View as matrix, grouping l as rows, (s,r) as columns: Dl x (2*Dr).
  Eigen::MatrixXcd right_mat() const {
    return Eigen::Map<const RowMat>(data.data(), Dl, 2 * Dr);
  }

  static Tensor from_left(const Eigen::MatrixXcd &M) { // rows = Dl*2, cols = Dr
    Tensor t(int(M.rows() / 2), int(M.cols()));
    Eigen::Map<RowMat>(t.data.data(), M.rows(), M.cols()) = M;
    return t;
  }
  static Tensor
  from_right(const Eigen::MatrixXcd &M) { // rows = Dl, cols = 2*Dr
    Tensor t(int(M.rows()), int(M.cols() / 2));
    Eigen::Map<RowMat>(t.data.data(), M.rows(), M.cols()) = M;
    return t;
  }
};

// Matrix Product State: psi(s0,s1,...,sN-1) = A0[s0] * A1[s1] * ... *
// AN-1[sN-1] where each A[s] is a Dl x Dr matrix. Edge bonds have dimension 1,
// so the full product is a 1x1 matrix = one amplitude.
struct MPS {
  int N;
  std::vector<Tensor> sites;
  // Truncation bookkeeping
  // discarded_weights[k] = fraction of ||theta||^2 thrown away at the k-th 2q
  // gate.
  std::vector<double> discarded_weights;
  // Fidelity estimates: prod(1 - w). Valid when state is in canonical form
  double fidelity = 1.0;

  // Orthogonality center. Sites < center are left-isometries, sites > center
  // are right-isometries. Product state: every site is trivially both.
  // 1q gates keep this: a unitary on s preserves the isometry sums.
  int center = 0;

  // A = Q R. Q becomes site i (left-isometry), R is pushed into site i+1.
  // State unchanged: A_i A_{i+1} = Q (R A_{i+1}).
  void left_canon(int i) {
    Eigen::MatrixXcd M = sites[i].left_mat();
    const int k = std::min<int>(M.rows(), M.cols());
    Eigen::HouseholderQR<Eigen::MatrixXcd> qr(M);
    Eigen::MatrixXcd Q =
        qr.householderQ() * Eigen::MatrixXcd::Identity(M.rows(), k);
    Eigen::MatrixXcd R =
        qr.matrixQR().topRows(k).triangularView<Eigen::Upper>();
    sites[i] = Tensor::from_left(Q);
    sites[i + 1] = Tensor::from_right(R * sites[i + 1].right_mat());
  }

  // Mirror image. Want M = L Qr with Qr having orthonormal rows.
  // QR of M^H = Q R  =>  M = R^H Q^H.  Q^H becomes site i, R^H goes into site
  // i-1.
  void right_canon(int i) {
    Eigen::MatrixXcd Mh = sites[i].right_mat().adjoint();
    const int k = std::min<int>(Mh.rows(), Mh.cols());
    Eigen::HouseholderQR<Eigen::MatrixXcd> qr(Mh);
    Eigen::MatrixXcd Q =
        qr.householderQ() * Eigen::MatrixXcd::Identity(Mh.rows(), k);
    Eigen::MatrixXcd R =
        qr.matrixQR().topRows(k).triangularView<Eigen::Upper>();
    sites[i] = Tensor::from_right(Q.adjoint());
    sites[i - 1] = Tensor::from_left(sites[i - 1].left_mat() * R.adjoint());
  }

  // Sweep the center to site c. Cost: one QR per site moved.
  void move_center(int c) {
    while (center < c) {
      left_canon(center);
      ++center;
    }
    while (center > c) {
      right_canon(center);
      --center;
    }
  }

  // With center set, the whole-state norm is the norm of one tensor.
  double norm() const {
    double s = 0;
    for (const cplx &a : sites[center].data)
      s += std::norm(a);
    return std::sqrt(s);
  }
  void normalize() {
    const double n = norm();
    for (cplx &a : sites[center].data)
      a /= n;
  }

  // Item 15: entanglement entropy (bits) across bond b (between sites b and
  // b+1). With center = b, singular values of A_b as a (Dl*2) x Dr matrix are
  // the Schmidt coefficients of the cut. p_k = s_k^2, S = -sum p log2 p.
  double entropy(int b) {
    move_center(b);
    Eigen::BDCSVD<Eigen::MatrixXcd> svd(sites[b].left_mat()); // values only
    Eigen::VectorXd p = svd.singularValues().array().square();
    p /= p.sum();
    double S = 0;
    for (int k = 0; k < p.size(); ++k)
      if (p(k) > 1e-15)
        S -= p(k) * std::log2(p(k));
    return S;
  }
  std::vector<double> entropies() {
    std::vector<double> S;
    for (int b = 0; b + 1 < N; ++b)
      S.push_back(entropy(b));
    return S;
  }

  std::vector<int> bond_dims() const { // N-1 internal bonds
    std::vector<int> d;
    for (int i = 0; i + 1 < N; ++i)
      d.push_back(sites[i].Dr);
    return d;
  }

  // Start in |00...0>. All bonds are 1 (product state = no entanglement).
  // A[l=0][s=0][r=0] = 1, everything else 0.
  explicit MPS(int N_) : N(N_) {
    for (int i = 0; i < N; ++i) {
      Tensor t(1, 1);
      t.at(0, 0, 0) = 1.0;
      sites.push_back(t);
    }
  }

  // Apply a 2x2 gate U to one site. Only touches the physical index:
  //   A'[l][s'][r] = sum_s U[s'][s] * A[l][s][r]
  // Bond dimensions do not change. No truncation. Exact.
  void apply_1q(int site, const Mat2 &U) {
    Tensor &A = sites[site];
    for (int l = 0; l < A.Dl; ++l)
      for (int r = 0; r < A.Dr; ++r) {
        cplx a0 = A.at(l, 0, r), a1 = A.at(l, 1, r);
        A.at(l, 0, r) = U[0][0] * a0 + U[0][1] * a1;
        A.at(l, 1, r) = U[1][0] * a0 + U[1][1] * a1;
      }
  }

  // Apply a 4x4 gate to adjacent sites (i, i+1).
  //   chi_max : keep at most this many singular values (bond dimension cap)
  //   cutoff  : drop s_k if s_k / s_0 <= cutoff
  void apply_2q(int i, const Mat4 &U, int chi_max = 1 << 30,
                double cutoff = 1e-14) {
    move_center(i); // makes truncation optimal and 'fidelity' exact
    Tensor &A = sites[i];
    Tensor &B = sites[i + 1];
    const int Dl = A.Dl, Dm = A.Dr,
              Dr = B.Dr; // A.Dr == B.Dl == Dm (shared bond)

    // --- Item 8: theta[l][s1][s2][r] = sum_m A[l][s1][m] * B[m][s2][r] ---
    // Contracting the shared bond m merges the two sites. Cost: Dl*Dm*Dr*4.
    std::vector<cplx> theta(Dl * 4 * Dr, 0.0);
    auto T = [&](int l, int s1, int s2, int r) -> cplx & {
      return theta[((l * 2 + s1) * 2 + s2) * Dr + r];
    };
    for (int l = 0; l < Dl; ++l)
      for (int s1 = 0; s1 < 2; ++s1)
        for (int s2 = 0; s2 < 2; ++s2)
          for (int r = 0; r < Dr; ++r) {
            cplx sum = 0;
            for (int m = 0; m < Dm; ++m)
              sum += A.at(l, s1, m) * B.at(m, s2, r);
            T(l, s1, s2, r) = sum;
          }

    // --- Item 9: apply gate on the physical pair, and reshape into a matrix
    // --- theta'[l][a][b][r] = sum_{s1,s2} U[2a+b][2s1+s2] *
    // theta[l][s1][s2][r] Matrix M groups (l,a) as rows and (b,r) as columns.
    // Rows = everything left of the cut, columns = everything right of the cut.
    // Schmidt decomposition of the state across this cut is the SVD of M.
    Eigen::MatrixXcd M(Dl * 2, 2 * Dr);
    for (int l = 0; l < Dl; ++l)
      for (int a = 0; a < 2; ++a)
        for (int b = 0; b < 2; ++b)
          for (int r = 0; r < Dr; ++r) {
            cplx sum = 0;
            for (int s1 = 0; s1 < 2; ++s1)
              for (int s2 = 0; s2 < 2; ++s2)
                sum += U[2 * a + b][2 * s1 + s2] * T(l, s1, s2, r);
            M(l * 2 + a, b * Dr + r) = sum;
          }

    // --- Item 10: SVD. M = Um * diag(s) * Vm^H. s is real, sorted descending.
    // ---
    Eigen::BDCSVD<Eigen::MatrixXcd> svd(M, Eigen::ComputeThinU |
                                               Eigen::ComputeThinV);
    const Eigen::VectorXd &s = svd.singularValues();

    // --- Items 11, 12: truncate and record the error ---
    // ||theta||^2 = sum s_k^2. Discarded weight = (sum of dropped s_k^2) /
    // total. This is the infidelity this truncation adds (optimal, by
    // Eckart-Young).
    const double total = s.squaredNorm();
    int k = 0;
    while (k < s.size() && k < chi_max && (k == 0 || s(k) > cutoff * s(0)))
      ++k;
    const double kept = s.head(k).squaredNorm();
    const double w = s.tail(s.size() - k).squaredNorm() / total;
    discarded_weights.push_back(w);
    fidelity *= (1.0 - w);

    // Rescale kept values so ||theta|| is unchanged (truncation shrinks it).
    const double scale = std::sqrt(total / kept);

    // --- Item 13: split back into two site tensors with new bond dimension k
    // --- Left tensor  = Um          (columns orthonormal: left-isometry) Right
    // tensor = s * Vm^H    (singular values absorbed to the right)
    Eigen::MatrixXcd Vh = svd.matrixV().adjoint();
    Tensor newA(Dl, k), newB(k, Dr);
    for (int l = 0; l < Dl; ++l)
      for (int a = 0; a < 2; ++a)
        for (int m = 0; m < k; ++m)
          newA.at(l, a, m) = svd.matrixU()(l * 2 + a, m);
    for (int m = 0; m < k; ++m)
      for (int b = 0; b < 2; ++b)
        for (int r = 0; r < Dr; ++r)
          newB.at(m, b, r) = s(m) * scale * Vh(m, b * Dr + r);
    sites[i] = std::move(newA);
    sites[i + 1] = std::move(newB);

    center = i + 1; // singluar values were absorbed into the right tesnsor
  }

  // Two-qubit gate on any pair q1 != q2. Same convention as ExactSim::apply_2q:
  // U index = 2*s(q1) + s(q2).
  // Every SWAP and the gate itself is an apply_2q, so chi_max and cutoff apply
  // at each step and each step adds to discarded_weights / fidelity.
  void apply_2q_any(int q1, int q2, Mat4 U, int chi_max = 1 << 30,
                    double cutoff = 1e-14) {
    assert(q1 != q2);
    if (q1 > q2) { // make q1 the left qubit
      U = gates::swap_order(U);
      std::swap(q1, q2);
    }
    // Carry logical qubit q1 to site q2-1. Sites q1..q2-2 now hold the old
    // contents of q1+1..q2-1, shifted left by one.
    for (int k = q1; k < q2 - 1; ++k)
      apply_2q(k, gates::SWAP(), chi_max, cutoff);
    apply_2q(q2 - 1, U, chi_max, cutoff);
    // Undo in reverse order: everything returns to its original site.
    for (int k = q2 - 2; k >= q1; --k)
      apply_2q(k, gates::SWAP(), chi_max, cutoff);
  }

  // Contract the whole chain into a dense 2^N vector. Exponential cost.
  // Only for testing against exact results (N <= ~20).
  // Qubit 0 is the most significant bit of the basis index.
  std::vector<cplx> to_statevector() const {
    // cur has shape (2^k, D): rows = configs of first k qubits,
    // cols = current right bond. Start: 1 config, bond 1, amplitude 1.
    std::vector<cplx> cur = {1.0};
    int rows = 1, D = 1;
    for (int i = 0; i < N; ++i) {
      const Tensor &A = sites[i];
      int newD = A.Dr;
      std::vector<cplx> next(rows * 2 * newD, 0.0);
      // next[(row*2 + s)][r] = sum_l cur[row][l] * A[l][s][r]
      for (int row = 0; row < rows; ++row)
        for (int s = 0; s < 2; ++s)
          for (int r = 0; r < newD; ++r) {
            cplx sum = 0;
            for (int l = 0; l < D; ++l)
              sum += cur[row * D + l] * A.at(l, s, r);
            next[(row * 2 + s) * newD + r] = sum;
          }
      cur = std::move(next);
      rows *= 2;
      D = newD;
    }
    return cur; // D == 1 at the end
  }
  // Probability that qubit q is |1>. With center = q, left and right
  // environments are orthonormal, so p(s) = sum_{l,r} |A[l][s][r]|^2 / norm^2.
  double prob1(int q) {
    move_center(q);
    const Tensor &A = sites[q];
    double p1 = 0, tot = 0;
    for (int l = 0; l < A.Dl; ++l)
      for (int s = 0; s < 2; ++s)
        for (int r = 0; r < A.Dr; ++r) {
          const double w = std::norm(A.at(l, s, r));
          tot += w;
          if (s == 1)
            p1 += w;
        }
    return p1 / tot;
  }

  // Projective measurement of qubit q. Collapses the state. Returns 0 or 1.
  // Only the center tensor changes, so canonical form is kept.
  int measure(int q, std::mt19937_64 &rng) {
    const double p1 = prob1(q); // moves center to q
    std::uniform_real_distribution<double> u(0.0, 1.0);
    const int out = (u(rng) < p1) ? 1 : 0;
    Tensor &A = sites[q];
    for (int l = 0; l < A.Dl; ++l)
      for (int r = 0; r < A.Dr; ++r)
        A.at(l, 1 - out, r) = 0.0; // project
    normalize();
    return out;
  }

  // Perfect sampling: one bitstring from |psi|^2, no 2^N vector. O(N chi^2).
  // Does not collapse the state (it only changes the gauge: center -> 0).
  std::vector<int> sample(std::mt19937_64 &rng) {
    move_center(0); // sites 1..N-1 are right-isometries
    std::uniform_real_distribution<double> u(0.0, 1.0);
    std::vector<int> out(N);
    // v = (product of the chosen matrices so far), a row vector of size Dl.
    Eigen::VectorXcd v(1);
    v(0) = 1.0;
    for (int i = 0; i < N; ++i) {
      const Tensor &A = sites[i];
      // w[s] = v^T A[s]. Right environment is the identity, so
      // p(s_i = s | earlier outcomes) is proportional to ||w[s]||^2.
      Eigen::VectorXcd w[2] = {Eigen::VectorXcd::Zero(A.Dr),
                               Eigen::VectorXcd::Zero(A.Dr)};
      for (int s = 0; s < 2; ++s)
        for (int l = 0; l < A.Dl; ++l)
          for (int r = 0; r < A.Dr; ++r)
            w[s](r) += v(l) * A.at(l, s, r);
      const double p0 = w[0].squaredNorm(), p1 = w[1].squaredNorm();
      const int s = (u(rng) * (p0 + p1) < p0) ? 0 : 1;
      out[i] = s;
      v = w[s] / std::sqrt(s == 0 ? p0 : p1); // renormalize: keeps numbers O(1)
    }
    return out;
  }
  // <n_q> = prob1(q) for every site. Center sweeps 0 -> N-1: cheap.
  std::vector<double> densities() {
    std::vector<double> n(N);
    for (int q = 0; q < N; ++q)
      n[q] = prob1(q);
    return n;
  }
};
