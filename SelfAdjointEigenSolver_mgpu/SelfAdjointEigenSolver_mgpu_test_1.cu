#include <catch2/catch_test_macros.hpp>
#include "MatrixGPU"
#include "tests/generateRandomMatrix.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <random>
#include <sys/time.h>
#include <curand.h>

#ifdef FLOAT
using RealScalar = float;
#else
using RealScalar = double;
#endif
// using Scalar = std::complex<RealScalar>; // Passes the test for std::complex<RealScalar> after ~400 sec
using Scalar    = cuda::std::complex<RealScalar>;  //
using ScalarCPU = std::complex<RealScalar>;

static inline double getETtime() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

TEST_CASE("MatrixGPU_mgpu", "test") {
#ifdef EIGEN_USE_MKL_ALL
	std::cout << "EIGEN_USE_MKL_ALL is set" << std::endl;
#endif
	// GPU::MAGMA::get_controller();
	magma_init();
	int ngpus = 0;
	cuCHECK(cudaGetDeviceCount(&ngpus));
	std::cout << "#\t ngpus = " << ngpus << std::endl;

	// constexpr int dim = 14602;  // Dimension of the zero momentum sector for Spin systems with L = 18
	// constexpr int dim = 26214;  // Dimension of the zero-momentum & even-parity sector for Spin systems with L = 20
	constexpr int dim
	    = 52428;  // Dimension of the zero momentum sector for Spin systems with L = 20 (Requires MKL_ILP64 interface)

	Eigen::MatrixX<ScalarCPU> mat(dim, dim);
	GPU::internal::generateRandomMatrix(mat, dim);

	constexpr double precision = 1.0E-4;
	{
		std::cout << "## Enter point 1" << std::endl;
		double                                          T_diag = getETtime();
		GPU::SelfAdjointEigenSolver_mgpu<decltype(mat)> hsolver(ngpus, mat);
		T_diag                            = getETtime() - T_diag;
		Eigen::MatrixX<ScalarCPU> eigVecs = hsolver.eigenvectors();
		auto const diff = (mat * eigVecs - eigVecs * hsolver.eigenvalues().asDiagonal()).norm();
		REQUIRE(diff < precision);
		std::cout << "## Passed point 1.\t T_diag = " << T_diag << " (sec)" << std::endl;
	}
	{
		std::cout << "## Enter point 2" << std::endl;
		GPU::SelfAdjointEigenSolver_mgpu<decltype(mat)> hsolver;
		double                                          T_diag = getETtime();
		hsolver.compute(ngpus, mat);
		T_diag                            = getETtime() - T_diag;
		Eigen::MatrixX<ScalarCPU> eigVecs = hsolver.eigenvectors();
		auto const diff = (mat * eigVecs - eigVecs * hsolver.eigenvalues().asDiagonal()).norm();
		REQUIRE(diff < precision);
		std::cout << "## Passed point 2.\t T_diag = " << T_diag << " (sec)" << std::endl;
	}

	magma_finalize();
}