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

TEST(SymmetricMatrixTests, RepeatedEigenvalue) {

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

int main(int argc, char **argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
