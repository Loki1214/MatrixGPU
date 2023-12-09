#include "ComplexEigenSolverGPU.cuh"
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

	std::mt19937                         engine(0);
	std::normal_distribution<RealScalar> dist(0.0, 1.0);
	Eigen::MatrixX<Scalar>               mat;

	for(auto dim = dimMin; dim <= dimMax; dim += step) {
		double          start, elapsed = 0.0, copyTime = 0.0;
		Eigen::VectorXd error(Nsample);
		for(auto i = 0; i < Nsample; ++i) {
			mat = Eigen::MatrixX<Scalar>::NullaryExpr(
			    dim, dim, [&]() { return Scalar(dist(engine), dist(engine)); });

			start = getETtime();
			GPU::MatrixGPU< decltype(mat) > dmat(mat);
			copyTime += getETtime() - start;
			start = getETtime();
			GPU::ComplexEigenSolver< decltype(mat) > solver(mat);
			elapsed += getETtime() - start;

			auto const& eigvecs = solver.eigenvectors();
			error(i) = (mat * eigvecs - eigvecs * solver.eigenvalues().asDiagonal()).norm();
		}
		elapsed /= double(Nsample);
		copyTime /= double(Nsample);

		std::cout << std::setw(4) << dim << " " << std::scientific << elapsed << " " << error.mean()
		          << " " << error.maxCoeff() << " " << copyTime << std::endl;
	}

	return EXIT_SUCCESS;
}