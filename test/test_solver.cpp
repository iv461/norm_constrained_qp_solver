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

Scalar evaluate_objective(const Mat &C, const Vec &b, const Vec &x) {
    return Scalar(0.5) * x.transpose() * C * x - b.dot(x);
}

TEST(NCSSolverTests, Smoke) {
    /// Smoke test.
    Mat C;
    C << -1.07822383, -2.78673686, -1.23438251
      ,-2.78673686, 0.93347297, 0.54945616
      , -1.23438251, 0.54945616, -0.05524914;

    Vec b;
    b << -0.68618036, -0.29540059, -0.51183855;
    Scalar s = 1.;
    
    auto x_hat = ncs::solve_norm_constrained_qp(C, b, s);
    
    auto obj1 = evaluate_objective(C, b, x_hat);

    fmt::println("x_hat: {}, Objective: {}", fmt::streamed(x_hat.transpose()), obj1);

    /// Correct solution, obtained using pymanopt
    Vec real_opt{-0.81721938, -0.48254904, -0.3151173};
    EXPECT_NEAR((real_opt - x_hat).norm(), 0, 1e-3);
    
    fmt::println("real_opt: {}, Objective: {}", fmt::streamed(real_opt.transpose()), evaluate_objective(C, b, real_opt));
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

    Mat C = (Q * eigs.asDiagonal() * Q.transpose());

    Vec b = Vec::Zero();
    
    Scalar s = 1.;
    auto x_hat = solve_norm_constrained_qp(C, b, s);
    
    Vec smallest_eigenvector = Q.col(2);

    auto norm_diff = (smallest_eigenvector - x_hat).norm();
    EXPECT_NEAR(x_hat.norm(), s, 1e-3);
    EXPECT_NEAR(norm_diff, 0, 1e-3);
}

TEST(NCSSolverTests, ZeroC) {
    /// Test where the matrix C is zero. In this case the problem simplifies to a linear one.
    Mat C{Mat::Zero()};
    Vec b = Vec::Random();

    Scalar s = 1.;

    auto x_hat = solve_norm_constrained_qp(C, b, s);
    //fmt::println("x_hat: {}", fmt::streamed(x_hat));
    EXPECT_NEAR(x_hat.normalized().dot(b.normalized()), 1, 1e-3);
    EXPECT_NEAR(x_hat.norm(), s, 1e-3);
    
}

TEST(NCSSolverTests, ZeroCandB) {
    /// Test where the matrix C is zero and b is zero. In this case, the objective is always zero.
    Mat C{Mat::Zero()};
    Vec b{Vec::Zero()};

    Scalar s = 1.;

    auto x_hat = solve_norm_constrained_qp(C, b, s);
    
    /// Solution does not matter, it only needs to be feasible
    EXPECT_NEAR(x_hat.norm(), s, 1e-3);
}

TEST(NCSSolverTests, RankOneC) {
    /// Test where the matrix C has rank one, i.e. all eigenvalue
    /// Small spectru
}

TEST(NCSSolverTests, SmallSpectrum) {
    /// Test where all the eigenvalues of matrix C are very close to each other
    /// TODO implement
}

TEST(NCSSolverTests, HugeSpectrum) {
    /// Test where all the eigenvalues of matrix C are very close to each other
    /// TODO implement
}


TEST(NCSSolverTests, LagrangeMultiplierCloseToEigenvalue) {
    /// Test where the optimal lagrange multiplier is close to the smallest Eigenvalue. 
    /// This test is supposed to test the numerical stability of the root finding of the secular equation.
    /// The secular equation has poles at every eigenvalue, so the root is very close to one of the poles. 
    /// We therefore test here whether the rootfinding reliably finds a root close to the pole.
    Mat C;
    C << 0.16904076, -0.57421902,  1.08854251,
        -0.57421902, -1.21054522, -2.82741677,
        1.08854251, -2.82741677,  0.84150445;

    Vec b;
    b << -0.8726189, 0.10428207, -0.21986756;
    Scalar s = 1.;
    auto x_hat = ncs::solve_norm_constrained_qp(C, b, s);
    
    auto obj1 = evaluate_objective(C, b, x_hat);

    /// Correct solution, obtained using pymanopt
    Vec real_opt{-0.31101768,  0.76470583,  0.56435184};
    EXPECT_NEAR((real_opt - x_hat).norm(), 0, 1e-3);    
}

TEST(NCSSolverTests, Hard2) {
    /// Another instance with b orthogonal to one of the eigenvectors
    Eigen::Matrix2<Scalar> C;
    C <<  3.29193704,  0.22878861,
        0.22878861, -3.19193704;

    Eigen::Vector2<Scalar> b;
    b << -0.0199698, 0.56664822;
    Scalar s = 1.;

    auto x_hat = ncs::solve_norm_constrained_qp(C, b, s);

    /// Correct solution, obtained using pymanopt
    Eigen::Vector2<Scalar> real_opt{-0.03522011,  0.99937958};
    EXPECT_NEAR((real_opt - x_hat).norm(), 0, 1e-3);

    Scalar obj_opt = -2.1669999983139068;
    Scalar obj_hat = Scalar(0.5) * x_hat.transpose() * C * x_hat - b.dot(x_hat);
    EXPECT_NEAR(obj_opt, obj_hat, 1e-3);
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
