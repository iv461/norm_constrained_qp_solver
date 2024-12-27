#include <gtest/gtest.h>

#include <fmt/core.h>
#include <fmt/ostream.h>

#include <norm_constrained_qp_solver.hpp>

using namespace ncs;

using Scalar = double; 
using Mat = Eigen::Matrix3<Scalar>;
using Vec = Eigen::Vector3<Scalar>;

/*int main() {
    return 0;
}*/

Mat createRandomMatrixC(const Vec &eigenvalues) {
    Mat D = Mat::Random(3, 3);
    Eigen::FullPivHouseholderQR<Mat> qr(D);
    Mat Q = qr.matrixQ();
    return Q * eigenvalues.asDiagonal() * Q.transpose();
}

Scalar evaluate_objective(const Mat &C, const Vec &b, const Vec &x) {
    return Scalar(0.5) * x.transpose() * C * x - b.dot(x);
}


TEST(NCSSolverTests, Smoke) {

    Mat D = Mat::Random(3, 3);

    Eigen::FullPivHouseholderQR<Mat> qr(D);
    Mat Q = qr.matrixQ();

    Vec eigs{Vec::Zero()};
    eigs[0] = 1.7;
    eigs[1] = 1.8;
    eigs[2] = -0.3;

    Mat C = Q * eigs.asDiagonal() * Q.transpose();
    Vec b = Vec::Random();
    Scalar s = 1.;
    auto x_hat = solve_norm_constrained_qp(C, b, s);

    fmt::println("C:\n{}, b:\n{}, s: {}", fmt::streamed(C), fmt::streamed(b.transpose()), s);
    
    auto obj1 = evaluate_objective(C, b, x_hat);
    fmt::println("obj1: {}", obj1);

    /**
     
    // Verify symmetry
    ASSERT_TRUE(symmetricMatrix.isApprox(symmetricMatrix.transpose(), 1e-10));

    // Check eigenvalues
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(symmetricMatrix);
    ASSERT_TRUE(solver.info() == Eigen::Success);

    // Verify that two eigenvalues are approximately equal
    EXPECT_NEAR(solver.eigenvalues()[0], solver.eigenvalues()[1], 1e-10);
    EXPECT_DOUBLE_EQ(solver.eigenvalues()[2], 3.0);
     */
}

TEST(NCSSolverTests, OrthogonalbSmoke) {



    Vec eigs{Vec::Zero()};
    eigs[1] = 1.8;
    eigs[0] = 1.7;
    eigs[2] = -0.3;

    Mat D = Mat::Random(3, 3);
    Eigen::FullPivHouseholderQR<Mat> qr(D);
    Mat Q = qr.matrixQ();
    Mat C = Q * eigs.asDiagonal() * Q.transpose();

    Vec b = Vec::Random();
    b = Q.col(0).cross(b); // Make b orthogonal to one of the eigenvectors so that one of the d_i is zero    

    Scalar s = 1.;

    fmt::println("C:\n{}, b:\n{}, s: {}", fmt::streamed(C), fmt::streamed(b.transpose()), s);

    auto x_hat = solve_norm_constrained_qp(C, b, s);
    
    auto obj1 = evaluate_objective(C, b, x_hat);
    fmt::println("obj1: {}", obj1);
    /**
     
    // Verify symmetry
    ASSERT_TRUE(symmetricMatrix.isApprox(symmetricMatrix.transpose(), 1e-10));

    // Check eigenvalues
    Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> solver(symmetricMatrix);
    ASSERT_TRUE(solver.info() == Eigen::Success);

    // Verify that two eigenvalues are approximately equal
    EXPECT_NEAR(solver.eigenvalues()[0], solver.eigenvalues()[1], 1e-10);
    EXPECT_DOUBLE_EQ(solver.eigenvalues()[2], 3.0);
     */
}

TEST(NCSSolverTests, Zerob) {
    /// Test where b is the zero-vector. In this case the optimization problem simplifies to one without the linear term.
    /// This problem has the optimal solution of the eigenvector associated with the smallest eigenvalue.

    Vec eigs{Vec::Zero()};
    eigs[0] = 1.8;
    eigs[1] = 1.7;
    eigs[2] = -0.3;

    Mat D = Mat::Random(3, 3);
    Eigen::FullPivHouseholderQR<Mat> qr(D);
    Mat Q = qr.matrixQ();
    Mat C = -(Q * eigs.asDiagonal() * Q.transpose());

    Vec b = Vec::Zero();
    
    Scalar s = 1.;
    auto x_hat = solve_norm_constrained_qp(C, b, s);
    
    Vec smallest_eigenvector = Q.col(2);
    
    auto norm_diff = (smallest_eigenvector - x_hat).norm();
    fmt::println("norm_diff: {}", norm_diff);
    EXPECT_NEAR(norm_diff, 0, 1e-3);
}

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
