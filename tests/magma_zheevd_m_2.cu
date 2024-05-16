// Compile command
// nvcc -o magma_zheevd_m_test -Xcompiler -fopenmp $(pkg-config --cflags magma) magma_zheevd_m_2.cu $(pkg-config --libs magma)
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
	std::vector<Scalar> mat(dim * dim);
	std::vector<Scalar> eigVecs(dim * dim);
	std::vector<double> eigVals(dim);
	{
		std::cout << "## Preparing a matrix on CPU." << std::endl;
#pragma omp parallel
		{
			std::random_device               seed_gen;
			std::mt19937                     engine(seed_gen());
			std::normal_distribution<double> dist(0.0, 1.0);
#pragma omp for
			for(magma_int_t j = 0; j < mat.size(); ++j) {
				mat.at(j) = Scalar(dist(engine), dist(engine));
			}
		}
		std::cout << "## Generated random numbers." << std::endl;
		double norm = 0;
#pragma omp parallel for schedule(dynamic, 100) reduction(+ : norm)
		for(magma_int_t j = 0; j < dim; ++j) {
			mat.at(j + dim * j) = 2 * std::real(mat.at(j + dim * j));
			norm += std::norm(mat.at(j + dim * j));
			for(magma_int_t k = 0; k < j; ++k) {
				mat.at(j + dim * k) = mat.at(j + dim * k) + std::conj(mat.at(k + dim * j));
				mat.at(k + dim * j) = std::conj(mat.at(j + dim * k));
				norm += 2 * std::norm(mat.at(j + dim * k));
			}
		}
		norm = std::sqrt(norm);
#pragma omp parallel for
		for(magma_int_t j = 0; j < mat.size(); ++j) {
			mat.at(j) /= norm;
			eigVecs.at(j) = mat.at(j);
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
	std::vector<Scalar> res(dim * dim, 0);
#pragma omp parallel for
	for(magma_int_t j = 0; j < dim; ++j)
		for(magma_int_t k = 0; k < dim; ++k) {
			res.at(j + dim * k) = eigVecs.at(j + dim * k) * eigVals.at(k);
		}
	MKL_Complex16 const alpha = {1.0, 0};
	MKL_Complex16 const beta  = {-1.0, 0};
	zhemm("L", "U", &dim, &dim, &alpha, reinterpret_cast<MKL_Complex16 const*>(mat.data()), &dim,
	      reinterpret_cast<MKL_Complex16 const*>(eigVecs.data()), &dim, &beta,
	      reinterpret_cast<MKL_Complex16*>(res.data()), &dim);

	double error = 0;
#pragma omp parallel for reduction(+ : error)
	for(magma_int_t j = 0; j < res.size(); ++j) error += std::norm(res.at(j));
	error = std::sqrt(error);
	std::cout << "# error = " << error << std::endl;

	magma_finalize();
	return EXIT_SUCCESS;
}