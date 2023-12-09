#pragma once

#if __has_include(<mkl.h>)
	#ifndef MKL
		#define MKL
	#endif
	#ifndef EIGEN_USE_MKL_ALL
		#define EIGEN_USE_MKL_ALL
	#endif
#else
	#if __has_include(<Accelerate/Accelerate.h>)
		#ifndef ACCELERATE
			#define ACCELERATE
		#endif
		#ifndef EIGEN_USE_BLAS
			#define EIGEN_USE_BLAS
		#endif
	#endif
#endif

#if defined(NDEBUG) || defined(__CUDA_ARCH__)
	#define DEBUG(arg)
#else
	#define DEBUG(arg) (arg)
#endif

#ifndef cuCHECK
	#define cuCHECK(call)                                                                   \
		{                                                                                   \
			const cudaError_t error = call;                                                 \
			if(error != cudaSuccess) {                                                      \
				fprintf(stderr, "cuCHECK Error: %s:%d,  ", __FILE__, __LINE__);             \
				fprintf(stderr, "code:%d, reason: %s\n", error, cudaGetErrorString(error)); \
				assert(error == cudaSuccess);                                               \
			}                                                                               \
		};
#endif

#ifndef CUSOLVER_CHECK
	#define CUSOLVER_CHECK(err)                                                            \
		do {                                                                               \
			cusolverStatus_t err_ = (err);                                                 \
			if(err_ != CUSOLVER_STATUS_SUCCESS) {                                          \
				fprintf(stderr, "cusolver error %d at %s:%d\n", err_, __FILE__, __LINE__); \
				throw std::runtime_error("cusolver error");                                \
			}                                                                              \
		} while(0);
#endif
