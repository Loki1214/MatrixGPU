// Compile command
// nvcc -o magma_zheevd_m_checkWorkSize -I../ -Xcompiler -fopenmp $(pkg-config --cflags magma) magma_zheevd_m_checkWorkSize.cu $(pkg-config --libs magma)
#include <magma_v2.h>
#include <magma_operators.h>
#include <vector>
#include <iostream>
#include <random>
#include <complex>

int main(int argc, char** argv) {
	if(argc != 2) {
		std::cerr << "Usage: 1.(This) 2.(L)" << std::endl;
		return EXIT_FAILURE;
	}
	magma_init();

	int ngpus = 0;
	cudaGetDeviceCount(&ngpus);
	std::cout << "#\t ngpus = " << ngpus << std::endl;

	int const L = std::stoi(argv[1]);
	magma_int_t const dim = (1 << (L-1))/L;
	std::cout << "#\t dim = " << dim << std::endl;
	// constexpr magma_int_t           dim = 52428;
	std::vector<magmaDoubleComplex> mat(1);

	magma_int_t                     info = 0;
	std::vector<double>             eigVals(1);
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

	std::cout << "#\t info = " << info << std::endl;

	magma_int_t memSize = 0;
	memSize = dim * dim * sizeof(magmaDoubleComplex); // mat
	// memSize += dim * dim * sizeof(magmaDoubleComplex); // eigVecs
	memSize += dim * sizeof(double); // eigVals
	memSize += lwork * sizeof(magmaDoubleComplex); // work
	memSize += lrwork * sizeof(double); // rwork
	memSize += liwork * sizeof(magma_int_t); // iwork
	std::cout << "#\t memSize = " << memSize << " bytes" << std::endl;

	magma_finalize();
	return EXIT_SUCCESS;
}