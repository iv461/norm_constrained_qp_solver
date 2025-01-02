// Copyright (c) 2025 Ivo Ivanov. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
//
// Implementation of a solver for norm-contrained quadratic programs, optimized for small and dense
// matrices.
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

template <typename Mat, typename Vec>
void check_arguments_dimensions(const Mat &C, const Vec &b) {
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
    throw std::invalid_argument(
        fmt::format("The matrix C must be at least 2x2, but instead it has {} rows and {} columns",
                    C.rows(), C.cols()));
}

template <typename Mat, typename Vec>
void check_matrix_argument_values(const Mat &C, const Vec &b) {
  if (!C.array().isFinite().all())
    throw std::invalid_argument(fmt::format(
        "C must be finite, i.e. not contain NaN or Infinite values, but instead C is: {}",
        fmt::streamed(C)));
  if (!b.array().isFinite().all())
    throw std::invalid_argument(fmt::format(
        "b must be finite, i.e. not contain NaN or Infinite value, but instead b is: {}",
        fmt::streamed(b)));
  if (!C.isApprox(C.transpose())) {
    auto norm_diff = (C - C.transpose()).norm();
    throw std::invalid_argument(
        fmt::format("The matrix C must be symmetric, instead ||C - C^T||_2 is: {}", norm_diff));
  }
}

template <typename Scalar>
void check_argument_s(Scalar s) {
  if (!std::isfinite(s))
    throw std::invalid_argument(
        fmt::format("s must be finite, i.e. NaN or Infinite, but it is instead {}", s));
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

template <typename Mat, typename Vec, typename Scalar>
void check_arguments(const Mat &C, const Vec &b, Scalar s) {
  check_arguments_dimensions(C, b);
  check_matrix_argument_values(C, b);
  check_argument_s(s);
}

/// Find the root of the secular equation.
template <typename Scalar, int Dim>
Scalar solve_secular_equation(const Eigen::Vector<Scalar, Dim> &D,
                              const Eigen::Vector<Scalar, Dim> &d, Scalar s) {
  const int dims = D.size();
  const Eigen::Vector<Scalar, Dim> d_sq = d.array().square();

  const auto secular_eq = [&](Scalar x) {
    return (d_sq.array() / (D.array() - x).square()).sum() - s * s;
  };

  /// Now bracket the root. First, find out which is the leftmost pole, since the leftmost root
  /// is before the leftmost pole. If and only if d_i^2 is exactly zero, then this pole has vanished
  /// mathematically. It also vanishes numerically if it's width is less than
  /// 1 ULP, we will check this later.
  size_t i_leftmost_pole = 0;
  for (int i = (dims - 1); i >= 0; i--)
    if (d_sq[i] != 0) i_leftmost_pole = i;

  auto leftmost_pole = D[i_leftmost_pole];
  auto abs_d_max = std::abs(d[i_leftmost_pole]);

  /// Now compute the interval. In the following, we locate the root to maximum machine precision.
  Scalar next_float_before_pole = std::nextafter(leftmost_pole, leftmost_pole - Scalar(1));
  Scalar leftmost_root = 0;
  /// At the pole, the secular equation is always mathematically undefined (division by zero) and
  /// is always numerically infinite. If at the next float to the right the secular equation is
  /// already negative, we know that the root is between the pole and the next float, i.e. have
  /// reached maximum machine precision and cannot locate it more precisely.
  if (secular_eq(next_float_before_pole) <= 0) {
    leftmost_root = next_float_before_pole;
  } else {
    const Scalar root_interval_right_border = next_float_before_pole;
    const Scalar root_interval_left_border = next_float_before_pole - d.norm() / s;
    /// Now use bisection, it is guaranteed to converge to a root as long as the interval is
    /// correct.
    const Scalar ULP = std::numeric_limits<Scalar>::epsilon();
    const auto [root, required_iterations] =
        bisect<Scalar>(secular_eq, root_interval_left_border, root_interval_right_border, ULP, 80);
    leftmost_root = root;
  }
  return leftmost_root;
}

/// Computes the optimal solution given the optimal Lagrange multiplier l (root of the secular
/// equation) and given the eigenvalue decomposition (Q, D) of the matrix.
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
/// The matrix C should be symmetric (but not positive definite).
/// The algorithm is based on eigendecomposition followed by root-finding.

/// We implement the first method from [1] that uses explicit root-finding, which was described in
/// that paper to be the fastest and most accurate method. In this paper, the discussion of this
/// problem starts with Eq. (10).
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

  /// First, handle the edge-case where C is the zero matrix:
  if (C.norm() < Scalar(1e-8)) {
    if (b.norm() <
        Scalar(1e-8)) {  /// Check if b is also zero, otherwise we may divide by zero below
      return Vec::Ones(dims).normalized() * s;  /// If both b and C are zero, return an arbitrary
                                                /// feasible solution, for example ones.
    }
    return b.normalized() *
           s;  /// If b is not zero, the linear objective - b^T x is minimized by maximizing the
               /// dot-product b^T x, which is maximized if x has the same direction as b.
  }

  /// Do eigen-decomposition. Since A is symmetric, we use the solver for symmetric (=self-adjoint)
  /// matrices. Eigen outputs the eigenvalues sorted in increasing order, a precondition needed in
  /// the following.
  Eigen::SelfAdjointEigenSolver<Mat> es(C);
  Mat Q = es.eigenvectors();
  Vec D = es.eigenvalues();

  /// Handle the case where the optimal Lagrange multiplier is the smallest eigenvalue: This is only
  /// the case if all d_i are zero. (Which can only happen if b is the zero vector)
  Vec d = Q.transpose() * b;

  if ((d.array() == 0).all()) {
    fmt::println("All d_i are zero case");
    return Q.col(0);
  }

  /// Otherwise, continue with root-finding. If at least one d_i is non-zero, the secular equation
  /// always has a root.
  Scalar leftmost_root = solve_secular_equation(D, d, s);

  /// Finally, extract the optimal solution given the optimal Lagrange multiplier (the root of the
  /// secular equation)
  Vec x = extract_solution(Q, D, d, leftmost_root);
  return x;
}

}  // namespace ncs