#include <gtest/gtest.h>

#include <fmt/core.h>
#include <fmt/ostream.h>

#include <norm_constrained_qp_solver.hpp>

using namespace ncs;

using Scalar = double; 
using Mat = Eigen::Matrix3<Scalar>;
using Vec = Eigen::Vector3<Scalar>;

// Test suite for argument validation
class CheckArgumentsTest : public ::testing::Test {
protected:
    // Helper function to create a symmetric matrix
    Eigen::MatrixXd createSymmetricMatrix(int size, Scalar fillValue = 1.0) {
        Eigen::MatrixXd mat = Eigen::MatrixXd::Constant(size, size, fillValue);
        return 0.5 * (mat + mat.transpose()); // Make symmetric
    }
};

// Positive test: Valid arguments
TEST_F(CheckArgumentsTest, ValidArguments) {
    Eigen::MatrixXd C = createSymmetricMatrix(3, 2.0); // 3x3 symmetric matrix
    Eigen::VectorXd b(3);
    b << 1.0, 2.0, 3.0; // Matching size
    Scalar s = 1.0; // Positive scalar

    // Should not throw
    EXPECT_NO_THROW(solve_norm_constrained_qp(C, b, s));
}

// Negative test: Non-square matrix C
TEST_F(CheckArgumentsTest, NonSquareMatrixC) {
    Eigen::MatrixXd C(3, 2); // Non-square matrix
    C.setRandom();
    Eigen::VectorXd b(3); // Vector size doesn't matter here
    b.setRandom();
    Scalar s = 1.0;

    // Expect an invalid_argument exception
    EXPECT_THROW({
        solve_norm_constrained_qp(C, b, s);
    }, std::invalid_argument);
}

// Negative test: Mismatched dimensions between C and b
TEST_F(CheckArgumentsTest, DimensionMismatch) {
    Eigen::MatrixXd C = createSymmetricMatrix(3, 2.0); // 3x3 symmetric matrix
    Eigen::VectorXd b(2); // Mismatched size
    b.setRandom();
    Scalar s = 1.0;

    EXPECT_THROW({
        solve_norm_constrained_qp(C, b, s);
    }, std::invalid_argument);
}

// Negative test: C not symmetric
TEST_F(CheckArgumentsTest, NonSymmetricMatrixC) {
    Eigen::MatrixXd C = Eigen::MatrixXd::Random(3, 3); // Random non-symmetric matrix
    Eigen::VectorXd b(3);
    b.setRandom();
    Scalar s = 1.0;

    EXPECT_THROW({
        solve_norm_constrained_qp(C, b, s);
    }, std::invalid_argument);
}

// Negative test: C contains NaN values
TEST_F(CheckArgumentsTest, NaNInMatrixC) {
    Eigen::MatrixXd C = createSymmetricMatrix(3);
    C(1, 1) = std::numeric_limits<Scalar>::quiet_NaN(); // Introduce NaN
    Eigen::VectorXd b(3);
    b.setRandom();
    Scalar s = 1.0;

    EXPECT_THROW({
        solve_norm_constrained_qp(C, b, s);
    }, std::invalid_argument);
}

// Negative test: b contains NaN values
TEST_F(CheckArgumentsTest, NaNInVectorB) {
    Eigen::MatrixXd C = createSymmetricMatrix(3);
    Eigen::VectorXd b(3);
    b << 1.0, std::numeric_limits<Scalar>::quiet_NaN(), 3.0; // Introduce NaN
    Scalar s = 1.0;

    EXPECT_THROW({
        solve_norm_constrained_qp(C, b, s);
    }, std::invalid_argument);
}

// Negative test: Non-positive scalar s
TEST_F(CheckArgumentsTest, NonPositiveScalar) {
    Eigen::MatrixXd C = createSymmetricMatrix(3);
    Eigen::VectorXd b(3);
    b.setRandom();
    Scalar s = -1.0; // Negative scalar

    EXPECT_THROW({
        solve_norm_constrained_qp(C, b, s);
    }, std::invalid_argument);
}

// Edge case: s is very small
TEST_F(CheckArgumentsTest, VerySmallScalarWarning) {
    Eigen::MatrixXd C = createSymmetricMatrix(3);
    Eigen::VectorXd b(3);
    b.setRandom();
    Scalar s = std::numeric_limits<Scalar>::epsilon() * 32; // Very small but valid scalar
    EXPECT_NO_THROW(solve_norm_constrained_qp(C, b, s));
}

// Negative test: Matrix C smaller than 2x2
TEST_F(CheckArgumentsTest, MatrixTooSmall) {
    Eigen::MatrixXd C(1, 1); // 1x1 matrix
    Eigen::VectorXd b(1);
    Scalar s = 1.0;
    EXPECT_THROW({
        solve_norm_constrained_qp(C, b, s);
    }, std::invalid_argument);
}

Mat createRandomMatrixC(const Vec &eigenvalues) {
    Mat D = Mat::Random(3, 3);
    Eigen::FullPivHouseholderQR<Mat> qr(D);
    Mat Q = qr.matrixQ();
    return Q * eigenvalues.asDiagonal() * Q.transpose();
}

Scalar evaluate_objective(const Mat &C, const Vec &b, const Vec &x) {
    return Scalar(0.5) * x.transpose() * C * x - b.dot(x);
}


TEST(NCSSolverTests, ArgumentChecking) {
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

    /// We need a larger spectrum for this test
    eigs[0] = 3.3;
    eigs[1] = -0.3;
    eigs[2] = -3.2; 

    Mat D = Mat::Random(3, 3);
    Eigen::FullPivHouseholderQR<Mat> qr(D);
    Mat Q = qr.matrixQ();
    Mat C = Q * eigs.asDiagonal() * Q.transpose();

    Vec random_axis{.234976, .58736, -0.654};
    Vec b = Q.col(0).cross(random_axis); // Make b orthogonal to one of the eigenvectors so that one of the d_i is zero.


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

TEST(NCSSolverTests, LagrangeMultiplierCloseToEigenvalue) {
    /// Test where the optimal lagrange multiplier is close to the smallest Eigenvalue. 
    /// This test is supposed to test the numerical stability of the root finding of the secular equation.
    /// The secular equation has poles at every eigenvalue, so the root is close to one of the poles. 
    /// We therefore test here whether the rootfinding reliably finds a root close to the pole.
    
    /// TODO implement
}

TEST(NCSSolverTests, LargeScale) {
    /// Here we test with large dynamically-sized matrices
    
    /// TODO implement
}

/// TODO test everything with float and double 
/// TODO Test small fixed-sized matrices, i.e. 2-6, 20, 40 and 100 



int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
