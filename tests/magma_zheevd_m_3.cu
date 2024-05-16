// Compile command
// nvcc -o magma_zheevd_m_3_test -Xcompiler -fopenmp $(pkg-config --cflags magma) magma_zheevd_m_3.cu $(pkg-config --libs magma)
#define MKL_ILP64
#include "error.hpp"
#include <mkl.h>
#include <magma_v2.h>
#include <magma_operators.h>
#include <vector>
#include <iostream>
#include <random>
#include <complex>

using Scalar = std::complex<double>;

int main() {
	magma_init();

	int ngpus = 0;
	cudaGetDeviceCount(&ngpus);
	std::cout << "#\t ngpus = " << ngpus << std::endl;

	constexpr magma_int_t dim = 52428;
	std::cout << "# dim = " << dim << "\n"
	          << "# dim * dim = " << dim * dim << std::endl;

	Eigen::MatrixX<Scalar> mat(dim, dim);
	Eigen::MatrixX<Scalar> eigVecs(dim, dim);
	Eigen::VectorXd        eigVals(dim);
	{
		std::cout << "## Preparing a matrix on CPU." << std::endl;
#pragma omp parallel
		{
			std::random_device               seed_gen;
			std::mt19937                     engine(seed_gen());
			std::normal_distribution<double> dist(0.0, 1.0);
#pragma omp for
			for(magma_int_t j = 0; j < mat.size(); ++j) {
				mat(j) = Scalar(dist(engine), dist(engine));
			}
		}
		std::cout << "## Generated random numbers." << std::endl;
		double norm = 0;
#pragma omp parallel for schedule(dynamic, 100) reduction(+ : norm)
		for(magma_int_t j = 0; j < dim; ++j) {
			mat(j, j) = 2 * std::real(mat(j, j));
			norm += std::norm(mat(j, j));
			for(magma_int_t k = 0; k < j; ++k) {
				mat(j, k) = mat(j, k) + std::conj(mat(k, j));
				mat(k, j) = std::conj(mat(j, k));
				norm += 2 * std::norm(mat(j, k));
			}
		}
		norm = std::sqrt(norm);
#pragma omp parallel for
		for(magma_int_t j = 0; j < mat.size(); ++j) {
			mat(j) /= norm;
			eigVecs(j) = mat(j);
		}
		std::cout << "## norm = " << norm << std::endl;
		std::cout << "## Prepared a matrix on CPU." << std::endl;
	}

	magma_int_t                     info = 0;
	std::vector<magmaDoubleComplex> work(1);
	std::vector<double>             rwork(1);
	std::vector<magma_int_t>        iwork(1);
	magma_zheevd_m(ngpus, MagmaVec, MagmaLower, dim,
	               reinterpret_cast<magmaDoubleComplex*>(eigVecs.data()), dim, eigVals.data(),
	               work.data(), -1, rwork.data(), -1, iwork.data(), -1, &info);
	magma_int_t const lwork  = magma_int_t(real(work[0]));
	magma_int_t const lrwork = magma_int_t(rwork[0]);
	magma_int_t const liwork = iwork[0];
	std::cout << "#\t  lwork = " << lwork << "\n"
	          << "#\t lrwork = " << lrwork << "\n"
	          << "#\t liwork = " << liwork << std::endl;
	work.resize(lwork);
	rwork.resize(lrwork);
	iwork.resize(liwork);
	magma_zheevd_m(ngpus, MagmaVec, MagmaLower, dim,
	               reinterpret_cast<magmaDoubleComplex*>(eigVecs.data()), dim, eigVals.data(),
	               work.data(), lwork, rwork.data(), lrwork, iwork.data(), liwork, &info);
	std::cout << "#\t info = " << info << std::endl;

	// Verify the results
	double error = GPU::internal::diagError(mat, eigVecs, eigVals);
	std::cout << "# error = " << error << std::endl;

	magma_finalize();
	return EXIT_SUCCESS;
}