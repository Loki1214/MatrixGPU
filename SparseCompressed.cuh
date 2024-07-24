#pragma once

#include "debug.hpp"
#include "ObjectOnGPU.cuh"
#include <Eigen/Sparse>
#include <thrust/device_vector.h>

template<typename Scalar, typename Index = Eigen::Index>
class SparseMatrix {
	public:
		Index                 m_outerSize;
		Index                 m_innerSize;
		Index                 m_nnz;  // number of non zeros
		Eigen::ArrayX<Index>  m_outerIndex;
		Eigen::ArrayX<Index>  m_innerIndex;
		Eigen::ArrayX<Scalar> m_values;

	public:
		__host__ __device__ Index         cols() const { return m_outerSize; }
		__host__ __device__ Index         rows() const { return m_innerSize; }
		__host__ __device__ Index*        outerIndexPtr() { return m_outerIndex.data(); }
		__host__ __device__ Index const*  outerIndexPtr() const { return m_outerIndex.data(); }
		__host__ __device__ Index*        innerIndexPtr() { return m_innerIndex.data(); }
		__host__ __device__ Index const*  innerIndexPtr() const { return m_innerIndex.data(); }
		__host__ __device__ Scalar*       valuePtr() { return m_values.data(); }
		__host__ __device__ Scalar const* valuePtr() const { return m_values.data(); }
};

template<typename Scalar, typename Index>
__global__ void construct_SparseMatrix_kernel(SparseMatrix<Scalar, Index>* dptr, Index outerSize,
                                              Index innerSize, Index nnz, Index* douterIndex,
                                              Index* dinnerIndex, Scalar* dvalues) {
	DEBUG(printf("# %s\n", __PRETTY_FUNCTION__));
	new(dptr) SparseMatrix<Scalar, Index>();
	dptr->m_outerSize = outerSize;
	dptr->m_innerSize = innerSize;
	dptr->m_nnz       = nnz;
	// for(Index j = 0; j <= outerSize; ++j) printf("#\t douterIndex[%d] = %d\n", int(j), int(douterIndex[j]));
	// dptr->m_outerIndex.resize(outerSize + 1);
	// dptr->m_innerIndex.resize(nnz);
	// dptr->m_values.resize(nnz);
	// for(Index j = 0; j <= outerSize; ++j) dptr->m_outerIndex(j) = douterIndex[j];
	dptr->m_outerIndex
	    = Eigen::Map<decltype(dptr->m_outerIndex)>(douterIndex, dptr->m_outerSize + 1);
	dptr->m_innerIndex = Eigen::Map<decltype(dptr->m_innerIndex)>(dinnerIndex, nnz);
	dptr->m_values     = Eigen::Map<decltype(dptr->m_values)>(dvalues, nnz);

	// for(Index j = 0; j <= outerSize; ++j) printf("%d ", int(dptr->m_outerIndex[j]));
	// printf("\n");
	// for(Index j = 0; j < nnz; ++j) printf("%d ", int(dptr->m_innerIndex[j]));
	// printf("\n");
	// for(Index j = 0; j < nnz; ++j) printf("(%lf, %lf) ", double(dptr->m_values[j].real()), double(dptr->m_values[j].imag()));
	// printf("\n");
}

template<typename Scalar, typename Index>
class ObjectOnGPU<SparseMatrix<Scalar, Index>> {
	private:
		using T = SparseMatrix<Scalar, Index>;
		T* m_ptr;

	public:
		ObjectOnGPU(Eigen::SparseMatrix<Scalar, 0, Index> const& spmat) {
			assert(spmat.isCompressed());
			// Eigen::SparseMatrix<std::complex<double>> spmat2(spmat.template cast<std::complex<double>>());
			// spmat2.makeCompressed();
			// std::cout << spmat2 << std::endl;

			thrust::device_vector<Index> outerIndices(
			    spmat.outerIndexPtr(), spmat.outerIndexPtr() + spmat.outerSize() + 1);
			thrust::device_vector<Index>  innerIndices(spmat.innerIndexPtr(),
			                                           spmat.innerIndexPtr() + spmat.nonZeros());
			thrust::device_vector<Scalar> values(spmat.valuePtr(),
			                                     spmat.valuePtr() + spmat.nonZeros());

			// std::cout << spmat.outerSize() << std::endl;
			// std::cout << spmat.nonZeros() << std::endl;
			// std::cout << "outerIndices.size() = " << outerIndices.size() << std::endl;
			// std::cout << "innerIndices.size() = " << innerIndices.size() << std::endl;
			// std::cout << "      values.size() = " << values.size() << std::endl;
			cuCHECK(cudaMalloc((void**)&m_ptr, sizeof(T)));
			construct_SparseMatrix_kernel<<<1, 1, 0, 0>>>(
			    m_ptr, spmat.outerSize(), spmat.innerSize(), spmat.nonZeros(),
			    outerIndices.data().get(), innerIndices.data().get(), values.data().get());
			cuCHECK(cudaGetLastError());
			cuCHECK(cudaDeviceSynchronize());
		}
		~ObjectOnGPU() {
			destruct_object_kernel<<<1, 1>>>(m_ptr);
			cuCHECK(cudaGetLastError());
			cuCHECK(cudaDeviceSynchronize());
			cuCHECK(cudaFree(m_ptr));
		}

		T*       ptr() { return m_ptr; }
		T const* ptr() const { return m_ptr; }
};