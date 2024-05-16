#include "macros.hpp"
#include "tests/generateRandomMatrix.hpp"
#include <Eigen/Dense>
#include <complex>
#include <iostream>
#include <iomanip>
#include <random>
#include <sys/time.h>

static inline double getETtime() {
	struct timeval tv;
	gettimeofday(&tv, NULL);
	return tv.tv_sec + (double)tv.tv_usec * 1e-6;
}

#ifdef FLOAT
using RealScalar = float;
#else
using RealScalar = double;
#endif
using Scalar = std::complex<RealScalar>;

int main(int argc, char** argv) {
#ifdef EIGEN_USE_MKL_ALL
	std::cout << "EIGEN_USE_MKL_ALL is set" << std::endl;
#endif
	if(argc < 5) {
		std::cerr << "Usage: 0.(This) 1.(dimMin) 2.(dimMax) 3.(step) 4.(Nsample)\n";
		std::cerr << "argc = " << argc << std::endl;
		std::exit(EXIT_FAILURE);
	}
	// std::cout << "# " << __FILE__ << std::endl;
	int const dimMin  = std::atoi(argv[1]);
	int const dimMax  = std::atoi(argv[2]);
	int const step    = std::atoi(argv[3]);
	int const Nsample = std::atoi(argv[4]);

	Eigen::initParallel();
	// std::mt19937                         engine(0);
	// std::normal_distribution<RealScalar> dist(0.0, 1.0);

	for(auto dim = dimMin; dim <= dimMax; dim += step) {
		Eigen::MatrixX<Scalar> mat(dim, dim);
		double                 start, elapsed = 0.0;
		Eigen::VectorXd        error(Nsample);
		for(auto i = 0; i < Nsample; ++i) {
			GPU::internal::generateRandomMatrix(mat, dim);

			start = getETtime();
			Eigen::SelfAdjointEigenSolver< decltype(mat) > solver(mat);
			elapsed += getETtime() - start;

			auto const& eigvecs = solver.eigenvectors();
			error(i) = (mat * eigvecs - eigvecs * solver.eigenvalues().asDiagonal()).norm();
		}
		elapsed /= double(Nsample);

		std::cout << std::setw(4) << dim << " " << std::scientific << elapsed << " " << error.mean()
		          << " " << error.maxCoeff() << std::endl;
	}

	return EXIT_SUCCESS;
}