#pragma once

#include <fmt/core.h>
#include <fmt/ostream.h>

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <cassert>
#include <exception>
#include <functional>
#include <tuple>

namespace ncs {

namespace {

template <typename Scalar>
std::pair<Scalar, size_t> bisect(std::function<Scalar(Scalar)> f, Scalar a, Scalar b, Scalar eps,
                                 size_t max_iterations) {
  Scalar c = a;
  size_t i = 0;
  if (f(a) * f(b) >= 0) {  /// Check for opposite sign
    throw std::invalid_argument(fmt::format(
        "The function must have opposite sign, but f({}) is {} and f({}) is {}", a, f(a), b, f(b)));
  }
  while ((b - a) >= eps && i < max_iterations) {
    i++;
    c = Scalar(.5) * (a + b);
    Scalar f_c = f(c);
    Scalar f_a = f(a);
    if (f_c == Scalar(0))
      return std::make_pair(c, i);
    else if (f_c * f_a < 0)
      b = c;
    else
      a = c;
  }
  return std::make_pair(c, i);
}

template <typename Scalar, int Dim>
void check_arguments(const Eigen::Matrix<Scalar, Dim, Dim> &C, const Eigen::Vector<Scalar, Dim> &b,
                     Scalar s) {
  static_assert(Dim > 1 || Dim == Eigen::Dynamic, "The matrix C must be at least 2x2");
  /// Check the dimensions if we are using dynamic-sized matrices
  if constexpr (Dim == Eigen::Dynamic) {
    if (C.cols() != C.rows())
      throw std::invalid_argument(
          fmt::format("The matrix C must be symmetric, but instead it has {} rows and {} columns",
                      C.rows(), C.cols()));
    if (b.size() != C.rows())
      throw std::invalid_argument(
          fmt::format("The vector b must have the same dimension as the matrix C, but C is of "
                      "dimension {} while b is of dimension {}",
                      C.rows(), b.size()));
    if (!(C.cols() > 1))
      throw std::invalid_argument(fmt::format(
          "The matrix C must be at least 2x2, but instead it has {} rows and {} columns", C.rows(),
          C.cols()));
  }
  if (!C.array().isFinite().all())
    throw std::invalid_argument(fmt::format(
        "C must be finite, i.e. not contain NaN or Infinite values, but instead C is: {}",
        fmt::streamed(C)));
  if (!b.array().isFinite().all())
    throw std::invalid_argument(fmt::format(
        "b must be finite, i.e. not contain NaN or Infinite value, but instead b is: {}",
        fmt::streamed(b)));
  if (!C.isApprox(C.transpose()))
    throw std::invalid_argument(fmt::format("The matrix C must be symmetric"));
  if (!(s > 0))
    throw std::invalid_argument(
        fmt::format("s must be a positive and non-zero, but it is instead {}", s));
  const auto s_min = std::numeric_limits<Scalar>::epsilon() * 32;
  if (s < s_min)
    throw std::invalid_argument(
        fmt::format("s is very small, an accurate result is not guaranteed, it must be greater or "
                    "equal {}, but instead it is: {}",
                    s_min, s));
}

template <typename Scalar, int Dim>
Scalar solve_secular_equation(const Eigen::Vector<Scalar, Dim> &D,
                              const Eigen::Vector<Scalar, Dim> &d, Scalar s) {
  const int dims = D.size();
  const Eigen::Vector<Scalar, Dim> d_sq = d.array().square();
  // fmt::println("d_sq: {}", fmt::streamed(d_sq) );

  const auto secular_eq = [&](Scalar x) {
    return (d_sq.array() / (D.array() - x).square()).sum() - s * s;
  };

  /// Now bracket the root. First, find out which is the right-most pole, since the right-most root
  /// comes after the last pole. If and only if d_i^2 is exactly zero, then this pole vanished
  /// mathematically (using real numbers). Numerically, it also vanishes if it's width is less than
  /// 1 ULP, we well check this later.
  size_t i_leftmost_pole = 0;
  for (int i = (dims - 1); i >= 0; i--)
    if (d_sq[i] != 0) i_leftmost_pole = i;

  // fmt::println("i_leftmost_pole: {}", i_leftmost_pole);

  auto last_pole = D[i_leftmost_pole];
  auto abs_d_max = std::abs(d[i_leftmost_pole]);

  /// Now compute the interval. In the following, we locate the root to maximum machine precision.
  Scalar next_float_before_pole = std::nextafter(last_pole, last_pole - Scalar(1));
  Scalar leftmost_root = 0;
  /// At the pole, the secular equation is mathematically always undefined (division by zero) and
  /// numerically always infinite. If at the next float to the right the secular equation is already
  /// negative, we know that the root is located between the pole and the next float, i.e. we
  /// reached maximum machine precision and cannot locate it more accurately.
  if (secular_eq(next_float_before_pole) <= 0) {
    leftmost_root = next_float_before_pole;
  } else {
    const Scalar root_interval_right_border = next_float_before_pole;
    const Scalar root_interval_left_border = next_float_before_pole - d.norm() / s;

    // fmt::println("root bracket: [{}, {}]", root_interval_left_border,
    // root_interval_right_border);

    /// Now use bisection, it is guaranteed to converge to a root as long as the interval is
    /// correct.
    const Scalar ULP = std::numeric_limits<Scalar>::epsilon();
    const auto [root, required_iterations] =
        bisect<Scalar>(secular_eq, root_interval_left_border, root_interval_right_border, ULP, 80);
    leftmost_root = root;
    /*
    fmt::print("Root-finding took {} iterations and found the Lagrange multiplier {} where the
    function is: {}\n", required_iterations, root, secular_eq(root));
    */
  }
  return leftmost_root;
}

/// Computes the optimal solution given the optimal Lagrange multiplier l (root of the secular
/// equation) and given the eigenvalue decomposition Q, D of the matrix
template <typename Scalar, int Dim>
Eigen::Vector<Scalar, Dim> extract_solution(const Eigen::Matrix<Scalar, Dim, Dim> &Q,
                                            const Eigen::Vector<Scalar, Dim> &D,
                                            const Eigen::Vector<Scalar, Dim> &d, Scalar l) {
  using Vec = Eigen::Vector<Scalar, Dim>;
  const Vec denom = (D.array() - l);
  Vec f{Vec::Zero(D.size())};
  for (int i = 0; i < D.size(); i++)
    if (denom[i] != 0)  /// Prevent division by zero
      f[i] = d[i] / denom[i];
    else
      f[i] = 0;  /// as required by the KKT conditions, Eq. (13) in [1]

  return Q * f;
}
}  // namespace

/// Solves the following non-convex optimization problem to global optimality:
///   min       1/2 x^T C x - b^T x
/// x \in R^d
///
/// subject to  ||x|| = s
///
/// The matrix C should be symmetric (but it does not need to be positive definite).
/// The algorithm is based on eigen-decomposition with subsequent root-finding.

/// We implement the first method from [1] that uses explicit root-finding which was evaluated in
/// the paper to be the fastest and most accurate method. In this paper, the discussion about this
/// problem starts at Eq. (10).
///
/// References:
/// [1] "A constrained eigenvalue problem", Walter Gander, Gene H. Golub, Urs von Matt,
/// https://doi.org/10.1016/0024-3795(89)90494-1
template <typename Scalar, int Dim>
static Eigen::Vector<Scalar, Dim> solve_norm_constrained_qp(
    const Eigen::Matrix<Scalar, Dim, Dim> &C, const Eigen::Vector<Scalar, Dim> &b, Scalar s) {
  using Mat = Eigen::Matrix<Scalar, Dim, Dim>;
  using Vec = Eigen::Vector<Scalar, Dim>;
  auto dims = C.cols();
  check_arguments(C, b, s);

  /// Edge-case where C is the zero-matrix:
  if (C.norm() < Scalar(1e-8)) {
    if (b.norm() < Scalar(1e-8)) {  /// Check if b is also zero, otherwise we may divide by zero
      return Vec::Ones(dims).normalized() * s;  /// If both b and C are zero, return an arbitrary
                                                /// feasible solution, for example ones.
    }
    return b.normalized() *
           s;  /// In this case, the linear objective - b^T x is minimized by maximizing the
               /// dot-product b^T x, which is maximized if x has the same direction as b.
  }

  /// Since A is symmetric, t we use solver for symmetric (=self-adjoint) matrices. Eigen outputs
  /// the eigenvalues sorted in increasing order, a precondition needed in the following.
  Eigen::SelfAdjointEigenSolver<Mat> es(C);
  Mat Q = es.eigenvectors();
  Vec D = es.eigenvalues();
  // fmt::println("Q: {}\nD: {}\nd: {}", fmt::streamed(Q), fmt::streamed(D.transpose()),
  // fmt::streamed(d.transpose()));

  /// Case analysis whether the optimal Lagrange multiplier is the smallest eigenvalue: This is only
  /// the case if all d_i are zero. (Which can only happen if b is the zero vector)
  Vec d = Q.transpose() * b;
  if ((d.array() == 0).all()) {
    return Q.col(0);
  }

  /// Otherwise, if at least one d_i is non-zero, the following secular equation always has a root.
  /// And the optimal Lagrange multiplier must be the root of it. (i.e. it must satisfy the KKT
  /// conditions, Eq. (13) in [1]) So we continue with root-finding.
  Scalar leftmost_root = solve_secular_equation(D, d, s);

  Vec x = extract_solution(Q, D, d, leftmost_root);
  return x;
}
}  // namespace ncs