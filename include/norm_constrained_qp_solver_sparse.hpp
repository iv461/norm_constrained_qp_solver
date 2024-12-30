
/// Experimental implementation for sparse matrices using SPECTRA. 
#pragma once

#include <fmt/core.h>
#include <fmt/ostream.h>

#include <Eigen/Core>
#include <Spectra/GenEigsSolver.h>
#include <Spectra/MatOp/DenseGenMatProd.h>
#include <norm_constrained_qp_solver.hpp> /// For check_arguments

namespace ncs {

namespace {

template <typename T>
static int sgn(T val) {
    return (T(0) < val) - (val < T(0));
}

}  // namespace

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

template <typename Scalar, int Dim>
static Eigen::Vector<Scalar, Eigen::Dynamic> algorithm2(
    const Eigen::Matrix<Scalar, Dim, Dim> &C, const Eigen::Vector<Scalar, Dim> &b, Scalar s) {
  using Mat = Eigen::Matrix<Scalar, Dim, Dim>;
  using Vec = Eigen::Vector<Scalar, Dim>;
  using MatD = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
  using VecD = Eigen::Vector<Scalar, Eigen::Dynamic>;

  auto dims = C.cols();
  
  check_arguments(C, b, s);

  /// Solve via implicit eigenproblem, construct matrix M
  MatD M{MatD::Zero(dims*2, dims*2)};

  M.block(0, 0, dims, dims) = -C;
  M.block(dims, dims, dims, dims) = -C;

  M.block(dims, 0, dims, dims) = MatD::Identity(dims, dims);
  M.block(0, dims, dims, dims) = (b * b.transpose()) / (s * s);
  //fmt::println("M:\n{}", fmt::streamed(M));


  // Construct matrix operation object using the wrapper class DenseSymMatProd
  Spectra::DenseGenMatProd<Scalar> op(M);
 
  // Construct eigen solver object, requesting the largest eigenvector only
  Spectra::GenEigsSolver<Spectra::DenseGenMatProd<Scalar>> eigs(op, 1, 6);
    
  // Initialize and compute
  eigs.init();
  
  int nconv = eigs.compute(Spectra::SortRule::LargestReal, 
    /*maxit=*/1000,
    /*tol=*/1e-10, 
    /*sorting=*/Spectra::SortRule::LargestReal
    );
  
  fmt::println("Solver result: nconv: {}, num_iterations: {}", nconv, eigs.num_iterations());
  
  VecD x_hat = VecD::Zero(dims);
  if (eigs.info() != Spectra::CompInfo::Successful) {
    fmt::println("Solver failed !");
    return x_hat;
  }

  /// Get largest eigenvector
  VecD z_star = eigs.eigenvectors(1).col(0).real();

  fmt::println("z_star: {}", fmt::streamed(z_star.transpose()));

  VecD z1_star = z_star.head(dims);
  VecD z2_star = z_star.tail(dims);

  fmt::println("z1_star:\n{}, z2_star:\n{}", fmt::streamed(z1_star.transpose()),
       fmt::streamed(z2_star.transpose()));

  Scalar g_times_z2 = b.dot(z2_star);

  Scalar z1_star_norm = z1_star.norm();
  if(z1_star_norm < Scalar(1e-8)) {
    fmt::println("Hard case occured, z1_star_norm is: {}", z1_star_norm);
  } 

  VecD x_opt = - Scalar(sgn(g_times_z2)) * s * (z1_star / z1_star_norm);
  /// Minus sign, since we have minus in front of the linear term of the objective.
  return -x_opt;
}
}  // namespace ncs