#include <catch2/catch_test_macros.hpp>
#include "MatrixGPU"
#include "tests/error.hpp"
#include "tests/generateRandomMatrix.hpp"
#include <Eigen/Dense>
#include <iostream>

#ifdef FLOAT
using RealScalar = float;
#else
using RealScalar = double;
#endif
using Scalar = std::complex<RealScalar>;

TEST_CASE("MatrixGPU", "test") {
	GPU::MAGMA::get_controller();

	constexpr int          dim = 100;
	Eigen::MatrixX<Scalar> mat(dim, dim);
	GPU::internal::generateRandomMatrix(mat, dim);
	// Eigen::ComplexEigenSolver<decltype(mat)> solver(mat);

	// GPU::MatrixGPU<decltype(mat)> dmat(mat);
	// std::cout << dmat << std::endl;
	GPU::ComplexEigenSolver<decltype(mat)> dsolver;
	dsolver.compute(mat);
	// std::cout << dsolver.eigenvalues() << std::endl;

	auto const diff = GPU::internal::diagError(mat, dsolver.eigenvectors(), dsolver.eigenvalues());
	std::cout << "# diff = " << diff << std::endl;

	constexpr double precision = 1.0E-4;
	REQUIRE(mat.rows() == dim);
	REQUIRE(mat.cols() == dim);
	REQUIRE(diff < precision);
}