// Compile command
// nvcc -o magma_zheevd_m_test -I../ -Xcompiler -fopenmp $(pkg-config --cflags magma) magma_zheevd_m.cu $(pkg-config --libs magma)
#include <magma_v2.h>
#include <magma_operators.h>
#include <vector>
#include <iostream>
#include <random>
#include <complex>

int main() {
	magma_init();

	int ngpus = 0;
	cudaGetDeviceCount(&ngpus);
	std::cout << "#\t ngpus = " << ngpus << std::endl;

	constexpr magma_int_t           dim = 52428;
	std::vector<magmaDoubleComplex> mat(dim * dim);
	{
		std::cout << "## Preparing a matrix on CPU." << std::endl;
#pragma omp parallel
		{
			std::random_device               seed_gen;
			std::mt19937                     engine(seed_gen());
			std::normal_distribution<double> dist(0.0, 1.0);
#pragma omp for
			for(magma_int_t j = 0; j < mat.size(); ++j) {
				mat.at(j) = MAGMA_Z_MAKE(dist(engine), dist(engine));
			}
		}
		std::cout << "## Generated random numbers." << std::endl;
		double norm = 0;
#pragma omp parallel for schedule(dynamic, 100) reduction(+ : norm)
		for(magma_int_t j = 0; j < dim; ++j) {
			mat.at(j + dim * j) = MAGMA_Z_MAKE(2 * real(mat.at(j + dim * j)), 0);
			norm += real(mat.at(j + dim * j) * conj(mat.at(j + dim * j)));
			for(magma_int_t k = 0; k < j; ++k) {
				mat.at(j + dim * k) = mat.at(j + dim * k) + conj(mat.at(k + dim * j));
				mat.at(k + dim * j) = conj(mat.at(j + dim * k));
				norm += 2 * real(mat.at(j + dim * k) * conj(mat.at(j + dim * k)));
			}
		}
		norm = std::sqrt(norm);
#pragma omp parallel for
		for(magma_int_t j = 0; j < mat.size(); ++j) { mat.at(j) /= norm; }
		std::cout << "## Prepared a matrix on CPU." << std::endl;
	}

	magma_int_t                     info = 0;
	std::vector<double>             eigVals(dim);
	std::vector<magmaDoubleComplex> work(1);
	std::vector<double>             rwork(1);
	std::vector<magma_int_t>        iwork(1);
	magma_zheevd_m(ngpus, MagmaVec, MagmaLower, dim, mat.data(), dim, eigVals.data(), work.data(),
	               -1, rwork.data(), -1, iwork.data(), -1, &info);
	magma_int_t const lwork  = magma_int_t(real(work[0]));
	magma_int_t const lrwork = magma_int_t(rwork[0]);
	magma_int_t const liwork = iwork[0];
	std::cout << "#\t  lwork = " << lwork << "\n"
	          << "#\t lrwork = " << lrwork << "\n"
	          << "#\t liwork = " << liwork << std::endl;
	work.resize(lwork);
	rwork.resize(lrwork);
	iwork.resize(liwork);
	magma_zheevd_m(ngpus, MagmaVec, MagmaLower, dim, mat.data(), dim, eigVals.data(), work.data(),
	               lwork, rwork.data(), lrwork, iwork.data(), liwork, &info);
	std::cout << "#\t info = " << info << std::endl;

	magma_finalize();
	return EXIT_SUCCESS;
}