

#pragma once 

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
//#include <boost/math/tools/roots.hpp>
#include <functional>
#include <tuple> 

namespace ncs {

namespace internal {

template<typename Scalar>
std::pair<Scalar, size_t> bisect(std::function<Scalar(Scalar)> f, 
  Scalar a, Scalar b, Scalar eps, size_t max_iterations) {
 
    Scalar c = a;
    size_t i = 0;

    if (f(a) * f(b) >= 0) { /// Check for opposite sign
        /// TODO throw, 
        return std::make_pair(a, i);
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

}

/// Solves the following non-convex optimization problem to global optimality: 
///   min      1/2 x^T A x + g^T x
/// x \in R^d
/// 
/// subject to  ||x|| = 1
///
/// The matrix A should be symmetric (but it does not need to be positive definite). 
/// The algorithm is an active-set solver based on eigen-decomposition with subsequent root-finding. 

/// We implement the first method from [1] that uses explicit root-finding which was evaluated in the paper to be
/// the fastest and most accurate.
///
/// References:
/// [1] "A constrained eigenvalue problem", Walter Gander, Gene H. Golub, Urs von Matt, https://doi.org/10.1016/0024-3795(89)90494-1
template<typename Scalar, int Dim>
  static Eigen::Vector<Scalar, Dim> solve_norm_constrained_qp(const Eigen::Matrix<Scalar, Dim, Dim> &A,
        const Eigen::Vector<Scalar, Dim> &g) {
    size_t d = A.cols(); /// TODO assert
    using Mat = Eigen::Matrix<Scalar, Dim, Dim>;
    using Vec = Eigen::Vector<Scalar, Dim>;


    /// Since A is symmetric, t we use solver for symmetric (=self-adjoint) matrices
    Eigen::SelfAdjointEigenSolver<Mat> es;
    es.compute(A);
    Mat Q = es.eigenvectors();
    Vec D = es.eigenvalues();
    Vec a = Q.transpose() * g;

    Vec a_sq = a.array().square();

    /// The rational function of which we need to find the roots. (named secular equation in [1])
    const auto characteristic_poly = [&](Scalar x) {
      Vec denom = (D.array() - x);
      Scalar f_x = (a_sq.array() / denom.array().square()).sum() - Scalar(1.);
      return f_x;
    };

    /// Find the index of the max element
    size_t max_b_i = 0;
    for (int i = 1; i < 3; i++)
      if (D[i] > D[max_b_i]) max_b_i = i;

    auto b_max = D[max_b_i];
    auto abs_a_max = std::abs(a[max_b_i]);
    Scalar l_hat{0};  /// The optimal root

    const double a_min_eps = 3.5e-15;  /// Expect 48 bit accuracy
    if (abs_a_max < a_min_eps) {
      /// In this case, the solution is close to one of the eigenvectors, maybe handle this case. Not
      /// sure if it matters numerically.
      l_hat = b_max + a_min_eps;
    } else {
      /// Compute an initial guess for the root with the largest x:
      /// Analytically, we can show that this is always positive
      /// We use the starting value as the left border of the interval in
      /// in which the root may lie (the 'bracket' as it's apparently called)
      const double k_eps =
          1.43e-14;  /// 2^-46 rounded up, has to be larger  than a_min_eps such that the interval of
                    /// the root has a width. We aim for 2 ULP width.
      double x0 = b_max + std::max(k_eps, abs_a_max - k_eps);
      /// And as the right border, we can easily show this:
      const auto right_root_border = a.norm() + b_max + k_eps;
      /// Now use boost root finding routine: Note that bisection is guaranteed to converge to a
      /// root. Analytically, our interval where the root should lie can be shown to contain the root
      /// (and only one root). Therefore, this method is guaranteed to converge to the root.
      /*
      fmt::print("x0: {}, f(x0): {}, right_root_border: {}, f(right_root_border): {}\n",
          fmt::streamed(a.transpose()), fmt::streamed(D.transpose()), x0,
              characteristic_poly(x0), right_root_border, characteristic_poly(right_root_border));
      */
      const auto [largest_root, required_iterations] = internal::bisect<Scalar>(characteristic_poly, x0, right_root_border, 
          1e-10, 20);
      /*const auto largest_root = boost::math::tools::newton_raphson_iterate(
          characteristic_poly, x0, x0, right_root_border, num_digits, num_iterations_performed);*/
      /*
      fmt::print("Root-finding took {} iterations and found {} where the function is: {}\n",
      num_iterations_performed, largest_root, characteristic_poly(largest_root));
      */
      l_hat = largest_root;
      // fmt::print("Computing D: {}, l: {}\n", fmt::streamed(D), l_hat);
    }
    /// This expression is the same as Q (L - l I)^-1 Q^T g.
    /// Q * (L - l I)^-1 * a <=> Q * (a / (L - l)), i.e element-wise vector division.
    Vec n = Q * (a.array() / (D.array() - l_hat)).matrix();
    
    /// TODO
    if (n.norm() < 1e-6) {  /// We can't even normalize, the numerical error is arbitrarily large
      n = Vec::Ones(d);
    }

    //n.normalize();

    return n;
  }
}