#include <catch2/catch_test_macros.hpp>
#include "MatrixGPU"
#include <Eigen/Dense>
#include <iostream>
#include <random>

#ifdef FLOAT
using RealScalar = float;
#else
using RealScalar = double;
#endif
using Scalar = std::complex<RealScalar>; // Passes the test for std::complex<RealScalar> after ~400 sec
// using Scalar = cuda::std::complex<RealScalar>; //

TEST_CASE("MatrixGPU", "test") {
#ifdef EIGEN_USE_MKL_ALL
	std::cout << "EIGEN_USE_MKL_ALL is set" << std::endl;
#endif
	GPU::MAGMA::get_contoroller();

	std::random_device                   seed_gen;
	std::mt19937                         engine(seed_gen());
	std::normal_distribution<RealScalar> dist(0.0, 1.0);

	constexpr int          dim = 14602; // Dimension of the zero momentum sector for Spin systems with L = 18
	Eigen::MatrixX<Scalar> mat = Eigen::MatrixX<Scalar>::NullaryExpr(
	    dim, dim, [&]() { return Scalar(dist(engine), dist(engine)); });
	mat = (mat + mat.adjoint()).eval();
	mat /= std::sqrt(mat.norm());
	std::cout << "## Prepared a matrix on CPU." << std::endl;
	GPU::MatrixGPU<decltype(mat)> dmat(mat);
	REQUIRE(dmat.rows() == dim);
	REQUIRE(dmat.cols() == dim);

	constexpr double precision = 1.0E-4;
	{
		std::cout << "## Enter point 1" << std::endl;
		GPU::SelfAdjointEigenSolver<decltype(mat)> hsolver(mat);
		auto const&                                eigVecs = hsolver.eigenvectors();
		auto const diff = (mat * eigVecs - eigVecs * hsolver.eigenvalues().asDiagonal()).norm();
		REQUIRE(diff < precision);
		std::cout << "## Passed point 1" << std::endl;
	}
	{
		std::cout << "## Enter point 2" << std::endl;
		GPU::SelfAdjointEigenSolver<decltype(mat)> hsolver;
		hsolver.compute(mat);
		auto const& eigVecs = hsolver.eigenvectors();
		auto const  diff    = (mat * eigVecs - eigVecs * hsolver.eigenvalues().asDiagonal()).norm();
		REQUIRE(diff < precision);
		std::cout << "## Passed point 2" << std::endl;
	}
	{
		std::cout << "## Enter point 3" << std::endl;
		GPU::SelfAdjointEigenSolver<decltype(dmat)> dsolver(dmat);
		auto const&                                 eigVecs = dsolver.eigenvectors();
		auto const diff = (mat * eigVecs - eigVecs * dsolver.eigenvalues().asDiagonal()).norm();
		REQUIRE(diff < precision);
		std::cout << "## Passed point 3" << std::endl;
	}
	{
		std::cout << "## Enter point 4" << std::endl;
		GPU::SelfAdjointEigenSolver<decltype(dmat)> dsolver;
		dsolver.compute(dmat);
		auto const& eigVecs = dsolver.eigenvectors();
		auto const  diff    = (mat * eigVecs - eigVecs * dsolver.eigenvalues().asDiagonal()).norm();
		REQUIRE(diff < precision);
		std::cout << "## Passed point 4" << std::endl;
	}
}