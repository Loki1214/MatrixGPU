#include <catch2/catch_test_macros.hpp>
#include "MatrixGPU.cuh"
#include "ComplexEigenSolverGPU.cuh"
#include <Eigen/Dense>
#include <iostream>
#include <random>

#ifdef FLOAT
using RealScalar = float;
#else
using RealScalar = double;
#endif
using Scalar = std::complex<RealScalar>;

TEST_CASE("MatrixGPU", "test") {
	GPU::MAGMA::get_contoroller();

	std::mt19937                         engine(0);
	std::normal_distribution<RealScalar> dist(0.0, 1.0);

	constexpr int          dim = 100;
	Eigen::MatrixX<Scalar> mat = Eigen::MatrixX<Scalar>::NullaryExpr(
	    dim, dim, [&]() { return Scalar(dist(engine), dist(engine)); });
	// Eigen::ComplexEigenSolver<decltype(mat)> solver(mat);

	// GPU::MatrixGPU<decltype(mat)> dmat(mat);
	// std::cout << dmat << std::endl;
	GPU::ComplexEigenSolver<decltype(mat)> dsolver;
	dsolver.compute(mat);
	// std::cout << dsolver.eigenvalues() << std::endl;

	auto const& eigVecs = dsolver.eigenvectors();
	auto const  diff    = (mat * eigVecs - eigVecs * dsolver.eigenvalues().asDiagonal()).norm();

	constexpr double precision = 1.0E-10;
	REQUIRE(mat.rows() == dim);
	REQUIRE(mat.cols() == dim);
	REQUIRE(diff < precision);
}