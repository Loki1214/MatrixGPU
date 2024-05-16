// Compile command
// g++ -o intel_mkl_zheevd_test -std=c++17 -fopenmp -DADD_ -DMKL_ILP64 intel_mkl_zheevd.cpp -L/opt/intel/oneapi/mkl/2023.0.0/lib/intel64 -lmkl_intel_ilp64 -lmkl_intel_thread -lmkl_core -liomp5 -lpthread -lstdc++ -lm
// icpx -o intel_mkl_zheevd_test -DMKL_ILP64 -qmkl-ilp64=parallel -qopenmp intel_mkl_zheevd.cpp
#define MKL_ILP64
#include <mkl.h>
#include <mkl_lapack.h>
#include <vector>
#include <iostream>
#include <random>
#include <complex>

using Scalar = std::complex<double>;

int main() {
	// constexpr MKL_INT dim = (1 << 16)/16;
	constexpr MKL_INT dim = 52428;
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
			for(MKL_INT j = 0; j < mat.size(); ++j) {
				mat.at(j) = Scalar(dist(engine), dist(engine));
			}
		}
		std::cout << "## Generated random numbers." << std::endl;
		double norm = 0;
#pragma omp parallel for schedule(dynamic, 100) reduction(+ : norm)
		for(MKL_INT j = 0; j < dim; ++j) {
			mat.at(j + dim * j) = 2 * std::real(mat.at(j + dim * j));
			norm += std::norm(mat.at(j + dim * j));
			for(MKL_INT k = 0; k < j; ++k) {
				mat.at(j + dim * k) = mat.at(j + dim * k) + std::conj(mat.at(k + dim * j));
				mat.at(k + dim * j) = std::conj(mat.at(j + dim * k));
				norm += 2 * std::norm(mat.at(j + dim * k));
			}
		}
		norm = std::sqrt(norm);
#pragma omp parallel for
		for(MKL_INT j = 0; j < mat.size(); ++j) {
			mat.at(j) /= norm;
			eigVecs.at(j) = mat.at(j);
		}
		std::cout << "## norm = " << norm << std::endl;
		std::cout << "## Prepared a matrix on CPU." << std::endl;
	}

	MKL_INT              info = 0;
	std::vector<Scalar>  work(1);
	std::vector<double>  rwork(1);
	std::vector<MKL_INT> iwork(1);
	MKL_INT              lwork  = -1;
	MKL_INT              lrwork = -1;
	MKL_INT              liwork = -1;
	zheevd("V", "L", &dim, reinterpret_cast<MKL_Complex16*>(eigVecs.data()), &dim, eigVals.data(),
	       reinterpret_cast<MKL_Complex16*>(work.data()), &lwork, rwork.data(), &lrwork,
	       iwork.data(), &liwork, &info);
	lwork  = static_cast<MKL_INT>(std::real(work[0]));
	lrwork = static_cast<MKL_INT>(rwork[0]);
	liwork = iwork[0];
	std::cout << "#\t  lwork = " << lwork << "\n"
	          << "#\t lrwork = " << lrwork << "\n"
	          << "#\t liwork = " << liwork << std::endl;
	work.resize(lwork);
	rwork.resize(lrwork);
	iwork.resize(liwork);
	zheevd("V", "L", &dim, reinterpret_cast<MKL_Complex16*>(eigVecs.data()), &dim, eigVals.data(),
	       reinterpret_cast<MKL_Complex16*>(work.data()), &lwork, rwork.data(), &lrwork,
	       iwork.data(), &liwork, &info);
	std::cout << "#\t info = " << info << std::endl;

	// Verify the results
	std::vector<Scalar> res(dim * dim, 0);
#pragma omp parallel for
	for(MKL_INT j = 0; j < dim; ++j)
		for(MKL_INT k = 0; k < dim; ++k) {
		res.at(j + dim * k) = eigVecs.at(j + dim * k) * eigVals.at(k);
	}
	MKL_Complex16 const alpha = {1.0, 0};
	MKL_Complex16 const beta  = {-1.0, 0};
	zhemm("L", "U", &dim, &dim, &alpha, reinterpret_cast<MKL_Complex16 const*>(mat.data()), &dim,
	      reinterpret_cast<MKL_Complex16 const*>(eigVecs.data()), &dim, &beta,
	      reinterpret_cast<MKL_Complex16*>(res.data()), &dim);

	double error = 0;
#pragma omp parallel for reduction(+ : error)
	for(MKL_INT j = 0; j < res.size(); ++j) error += std::norm(res.at(j));
	error = std::sqrt(error);
	std::cout << "# error = " << error << std::endl;

	return EXIT_SUCCESS;
}