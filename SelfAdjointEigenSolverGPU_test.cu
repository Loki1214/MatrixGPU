#include <catch2/catch_test_macros.hpp>
#include "MatrixGPU.cuh"
#include "SelfAdjointEigenSolverGPU.cuh"
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

	constexpr int          dim = 1000;
	Eigen::MatrixX<Scalar> mat = Eigen::MatrixX<Scalar>::NullaryExpr(
	    dim, dim, [&]() { return Scalar(dist(engine), dist(engine)); });
	mat = (mat + mat.adjoint()).eval();
	mat /= std::sqrt(mat.norm());
	GPU::MatrixGPU<decltype(mat)> dmat(mat);
	REQUIRE(dmat.rows() == dim);
	REQUIRE(dmat.cols() == dim);

	constexpr double precision = 1.0E-4;
	{
		GPU::SelfAdjointEigenSolver<decltype(mat)> hsolver(mat);
		auto const&                                eigVecs = hsolver.eigenvectors();
		auto const diff = (mat * eigVecs - eigVecs * hsolver.eigenvalues().asDiagonal()).norm();
		REQUIRE(diff < precision);
	}
	{
		GPU::SelfAdjointEigenSolver<decltype(mat)> hsolver;
		hsolver.compute(mat);
		auto const& eigVecs = hsolver.eigenvectors();
		auto const  diff    = (mat * eigVecs - eigVecs * hsolver.eigenvalues().asDiagonal()).norm();
		REQUIRE(diff < precision);
	}
	{
		GPU::SelfAdjointEigenSolver<decltype(dmat)> dsolver(dmat);
		auto const&                                 eigVecs = dsolver.eigenvectors();
		auto const diff = (mat * eigVecs - eigVecs * dsolver.eigenvalues().asDiagonal()).norm();
		REQUIRE(diff < precision);
	}
	{
		GPU::SelfAdjointEigenSolver<decltype(dmat)> dsolver;
		dsolver.compute(dmat);
		auto const& eigVecs = dsolver.eigenvectors();
		auto const  diff    = (mat * eigVecs - eigVecs * dsolver.eigenvalues().asDiagonal()).norm();
		REQUIRE(diff < precision);
	}
}