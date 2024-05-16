#define EIGEN_DONT_PARALLELIZE
#include "SelfAdjointEigenSolver_mgpu.cuh"
#include "tests/generateRandomMatrix.hpp"
#include "tests/error.hpp"
#include <Eigen/Dense>
#include <complex>
#include <iostream>
#include <iomanip>

#ifdef FLOAT
using RealScalar = float;
#else
using RealScalar = double;
#endif
using Scalar = std::complex<RealScalar>;

int main(int argc, char** argv) {
#ifdef EIGEN_USE_MKL_ALL
	std::cout << "# EIGEN_USE_MKL_ALL is set." << std::endl;
#else
	std::cout << "# EIGEN_USE_MKL_ALL is NOT set." << std::endl;
#endif
	if(argc < 4) {
		std::cerr << "Usage: 0.(This) 1.(LMax) 2.(LMin) 3.(Nsample)\n";
		std::cerr << "argc = " << argc << std::endl;
		std::exit(EXIT_FAILURE);
	}
	GPU::MAGMA::get_controller();

	std::cout << "# " << __FILE__ << std::endl;
	int const LMax    = std::atoi(argv[1]);
	int const LMin    = std::atoi(argv[2]);
	int const Nsample = std::atoi(argv[3]);
	std::cout << "# LMax = " << LMax << ", LMin = " << LMin << ", Nsample = " << Nsample
	          << std::endl;

	for(auto L = LMin; L <= LMax; ++L) {
		Eigen::Index const dim = int(1 << L) / L;
		std::cout << "# L = " << L << ", dim = " << dim << std::endl;
		std::cout << "# dim * dim = " << dim * dim << std::endl;
		double                 start, Tdiag = 0.0;
		Eigen::VectorXd        error(Nsample);
		Eigen::MatrixX<Scalar> mat(dim, dim);
		for(auto i = 0; i < Nsample; ++i) {
			GPU::internal::generateRandomMatrix(mat, dim);
			start = GPU::internal::getETtime();
			GPU::SelfAdjointEigenSolver_mgpu<decltype(mat)> solver(GPU::MAGMA::ngpus(), mat);
			Tdiag += GPU::internal::getETtime() - start;
			start    = GPU::internal::getETtime();
			error(i) = GPU::internal::diagError(mat, solver.eigenvectors(), solver.eigenvalues());
			std::cout << "## error = " << error(i) << ", Tdiag = " << Tdiag << " (sec)"
			          << std::endl;
			std::cout << "## Calculated the error. T_err = " << GPU::internal::getETtime() - start
			          << " (sec)" << std::endl;
		}
		Tdiag /= double(Nsample);

		std::cout << std::setw(4) << dim << " " << std::scientific << Tdiag << " " << error.mean()
		          << " " << error.maxCoeff() << std::endl;
	}

	return EXIT_SUCCESS;
}