// Copyright (c) 2025 Ivo Ivanov. All rights reserved.
// Licensed under the MIT license. See LICENSE file in the project root for details.
//
// Experimental implementation of the sovler for sparse matrices using the SPECTRA library (similar to ARPACK).
#pragma once

#include <fmt/core.h>
#include <fmt/ostream.h>

#include <Eigen/Core>
#include <Eigen/Sparse>

#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/DenseGenMatProd.h>

#include <norm_constrained_qp_solver.hpp> /// For check_arguments
namespace ncs {

namespace {

template <typename T>
static int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

template <typename Mat, typename Vec, typename Scalar>
void check_arguments_sparse(const Mat &C, const Vec &b, Scalar s) {
  check_arguments_dimensions(C, b);
  check_argument_s(s);
}

// An matrix multiplication operator that implements the block-matrix multiplication required by algorithm, in form of the interface required by the SPECTRA library.
// It is templated on the matrix and vector operator to be able to use it matrix-free.
 template <typename Scalar_, typename MatrixFunc, typename VectorFunc>
class MatrixProd {
 public:
     using Scalar = Scalar_;
 private:
     using Vector = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;
     using MapConstVec = Eigen::Map<const Vector>;
     using MapVec = Eigen::Map<Vector>;
     const MatrixFunc& C_;
     const VectorFunc& b_;
     Scalar s_{0};
 public:
     MatrixProd(const MatrixFunc& C, const VectorFunc& b, Scalar s) : C_(C), b_(b), s_(s) {}
     auto rows() const { return 2 * C_.rows(); }
     auto cols() const { return 2 * C_.cols(); }
     // y_out = A * x_in
     void perform_op(const Scalar* x_in, Scalar* y_out) const {
         const int dims = C_.cols();
         MapConstVec x(x_in, 2* dims);
         MapVec y(y_out, 2*dims);

          /// Now compute y_out =
          /// [ -C  b * b^T/s^2 ] [x_h]
          /// [  I       -C     ] [x_t]
          /// 
        y.head(dims).noalias() = - C_ * x.head(dims) + b_ * b_.dot(x.tail(dims)) / (s_ * s_);
        y.tail(dims).noalias() = x.head(dims) - C_ * x.tail(dims); 
     }
 };

}  // namespace

/// Overload for sparse eigen matrices.
template <typename Scalar>
static Eigen::Vector<Scalar, Eigen::Dynamic> solve_norm_constrained_qp_sparse(
    const Eigen::SparseMatrix<Scalar> &C,
    const Eigen::SparseMatrix<Scalar> &b, 
    Scalar s) {
    check_arguments_sparse(C, b, s);
    return solve_norm_constrained_qp_sparse<Scalar>(C, b, s);
}

/// Solves the following non-convex optimization problem to global optimality:
///   min       1/2 x^T C x - b^T x
/// x \in R^d
///
/// subject to  ||x|| = s
///
/// The matrix C should be symmetric (but not positive definite).
/// The algorithm is optimized for large and/or sparse matrices.
///
/// References:
/// [1] "An active-set algorithm for norm constrained quadratic problems", Nikitas Rontsis, Paul J. Goulart & Yuji Nakatsukasa,
/// https://doi.org/10.1007/s10107-021-01617-2
template <typename Scalar, typename Vec, typename Mat>
static Eigen::Vector<Scalar, Eigen::Dynamic> solve_norm_constrained_qp_sparse(
    const Mat &C, const Vec &b, Scalar s) {
  using MatD = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using VecD = Eigen::Vector<Scalar, Eigen::Dynamic>;

  check_arguments_sparse(C, b, s);
  auto dims = C.cols();

  using MOp = MatrixProd<Scalar, Mat, Vec>;
  MOp op(C, b, s);
  /// Construct the eigensolver, 1 means "compute only one eigenvalue/eigenvector", whereas 6 is some parameter for controlling the convergence of the algorithm (6 is the default)
  Spectra::GenEigsSolver<MOp> eigs(op, 1, 6);
  // Initialize it, this chooses a random starting point.
  eigs.init();
  /// Now run the Arnoldi iteration algorithm.
  int nconv = eigs.compute(Spectra::SortRule::LargestReal, /*maxit=*/1000, /*tol=*/1e-10, /*sorting=*/Spectra::SortRule::LargestReal);

  VecD x_hat = VecD::Zero(dims);
  auto eigs_info = eigs.info();
  if (eigs_info != Spectra::CompInfo::Successful || nconv != 1) {
    throw std::invalid_argument(fmt::format("Solver failed to converge, eigs.info() was {}, nconv was {}", int(eigs_info), nconv));
  }
  /// Get largest eigenvector. The associated eigenvalue is always real (see Theorem 1. in [1] and the references therein)
  VecD z_star = eigs.eigenvectors(1).col(0).real();
  VecD z1_star = z_star.head(dims);
  VecD z2_star = z_star.tail(dims);
  Scalar z1_star_norm = z1_star.norm();
  Scalar g_times_z2 = b.dot(z2_star);
  /// Now check for the hard-case as defined in [1], page 6. Solving this hard case is currently not implemented.
  if(z1_star_norm < Scalar(1e-8)) {
    throw std::invalid_argument(fmt::format("The solver can't solve this instance (hard case occurred, where the norm of z1-star is close to zero (it was {}).", z1_star_norm));
  }
  VecD x_opt = - Scalar(sgn(g_times_z2)) * s * (z1_star / z1_star_norm);
  /// Minus sign, since we have minus in front of the linear term of the objective compared to [1]
  return -x_opt;
}
}  // namespace ncs