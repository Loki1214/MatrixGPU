#pragma once

#include <magma_v2.h>
#include <magma_operators.h>
#include <cuda/std/complex>

// MAGMA_COMPLEX_TYPES and CUDA_STD_COMPLEX_TYPES
#include <Eigen/Core>
namespace Eigen {

	template<>
	struct NumTraits<magmaFloatComplex> : GenericNumTraits<magmaFloatComplex> {
			typedef float                             Real;
			typedef typename NumTraits<Real>::Literal Literal;
			enum {
				IsComplex             = 1,
				RequireInitialization = NumTraits<Real>::RequireInitialization,
				ReadCost              = 2 * NumTraits<Real>::ReadCost,
				AddCost               = 2 * NumTraits<Real>::AddCost,
				MulCost               = 4 * NumTraits<Real>::MulCost + 2 * NumTraits<Real>::AddCost
			};

			EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR static inline Real epsilon() {
				return NumTraits<Real>::epsilon();
			}
			EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR static inline Real dummy_precision() {
				return NumTraits<Real>::dummy_precision();
			}
			EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR static inline int digits10() {
				return NumTraits<Real>::digits10();
			}
	};

	template<>
	struct NumTraits<magmaDoubleComplex> : GenericNumTraits<magmaDoubleComplex> {
			typedef double                            Real;
			typedef typename NumTraits<Real>::Literal Literal;
			enum {
				IsComplex             = 1,
				RequireInitialization = NumTraits<Real>::RequireInitialization,
				ReadCost              = 2 * NumTraits<Real>::ReadCost,
				AddCost               = 2 * NumTraits<Real>::AddCost,
				MulCost               = 4 * NumTraits<Real>::MulCost + 2 * NumTraits<Real>::AddCost
			};

			EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR static inline Real epsilon() {
				return NumTraits<Real>::epsilon();
			}
			EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR static inline Real dummy_precision() {
				return NumTraits<Real>::dummy_precision();
			}
			EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR static inline int digits10() {
				return NumTraits<Real>::digits10();
			}
	};

	template<typename Real_>
	struct NumTraits<cuda::std::complex<Real_> > : GenericNumTraits<cuda::std::complex<Real_> > {
			typedef Real_                              Real;
			typedef typename NumTraits<Real_>::Literal Literal;
			enum {
				IsComplex             = 1,
				RequireInitialization = NumTraits<Real_>::RequireInitialization,
				ReadCost              = 2 * NumTraits<Real_>::ReadCost,
				AddCost               = 2 * NumTraits<Real>::AddCost,
				MulCost               = 4 * NumTraits<Real>::MulCost + 2 * NumTraits<Real>::AddCost
			};

			EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR static inline Real epsilon() {
				return NumTraits<Real>::epsilon();
			}
			EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR static inline Real dummy_precision() {
				return NumTraits<Real>::dummy_precision();
			}
			EIGEN_DEVICE_FUNC EIGEN_CONSTEXPR static inline int digits10() {
				return NumTraits<Real>::digits10();
			}
	};
}  // namespace Eigen

// MAGMA_BLAS
template<typename Index_, typename Scalar_>
static inline void magma_axpy(Index_ n, Scalar_ alpha, Scalar_ const* dx, Index_ incx, Scalar_* dy,
                              Index_ incy, magma_queue_t queue) {
	if constexpr(std::is_same_v<Scalar_, cuda::std::complex<float>>) {
		magma_caxpy(n, alpha, dx, incx, dy, incy, queue);
	}
	else if constexpr(std::is_same_v<Scalar_, cuda::std::complex<double>>) {
		magma_zaxpy(n, alpha, dx, incx, dy, incy, queue);
	}
	else {
		static_assert([] { return false; }(),
		              "Error: magma_axpy for the input scalar type is NOT implemented.");
	}
}

// static inline magma_int_t magmablas_hemv_work
// (magma_uplo_t Uplo, magma_int_t n, Complex_t<float> alpha,
//   matrix_gpu<Complex_t<float>> const& dA,
//   Complex_t<float> const* dx, magma_int_t incx,  Complex_t<float> beta,
//   Complex_t<float>* dy,       magma_int_t incy,
//   Complex_t<float>* dwork,    magma_int_t lwork, magma_queue_t queue)
// {
//  return magmablas_chemv_work(Uplo, n, alpha, dA.ptr(), dA.LD(), dx, incx, beta, dy, incy, dwork, lwork, queue);
// }
//
// static inline magma_int_t magmablas_hemv_work
// (magma_uplo_t Uplo, magma_int_t n, Complex_t<double> alpha,
//   matrix_gpu<Complex_t<double>> const& dA,
//   Complex_t<double> const* dx, magma_int_t incx,  Complex_t<double> beta,
//   Complex_t<double>* dy,       magma_int_t incy,
//   Complex_t<double>* dwork,    magma_int_t lwork, magma_queue_t queue)
// {
//  return magmablas_zhemv_work(Uplo, n, alpha, dA.ptr(), dA.LD(), dx, incx, beta, dy, incy, dwork, lwork, queue);
// }

template<typename Index_, typename Scalar_>
static inline Scalar_ magma_dotc(Index_ n, Scalar_ const* dx, Index_ incx, Scalar_ const* dy,
                                 Index_ incy, magma_queue_t queue) {
	if constexpr(std::is_same_v<Scalar_, float>) {
		return magma_sdot(n, dx, incx, dy, incy, queue);
	}
	else if constexpr(std::is_same_v<Scalar_, double>) {
		return magma_ddot(n, dx, incx, dy, incy, queue);
	}
	else if constexpr(std::is_same_v<Scalar_, cuda::std::complex<float>>) {
		auto const res = magma_cdotc(n, reinterpret_cast<magmaFloatComplex const*>(dx), incx,
		                             reinterpret_cast<magmaFloatComplex const*>(dy), incy, queue);
		return Scalar_(real(res), imag(res));
	}
	else if constexpr(std::is_same_v<Scalar_, cuda::std::complex<double>>) {
		auto const res = magma_zdotc(n, reinterpret_cast<magmaDoubleComplex const*>(dx), incx,
		                             reinterpret_cast<magmaDoubleComplex const*>(dy), incy, queue);
		return Scalar_(real(res), imag(res));
	}
	else {
		static_assert([] { return false; }(),
		              "Error: magma_dotc for the input scalar type is NOT implemented.");
	}
	// To suppress wrong compiler warning "warning: missing return statement at end of non-void function"
	// Never be executed
	return 0;
}

template<typename Index_, typename Scalar_>
static inline void magma_hemm(magma_side_t side, magma_uplo_t uplo, Index_ m, Index_ n,
                              Scalar_ alpha, Scalar_ const* dA, Index_ ldda, Scalar_ const* dB,
                              Index_ lddb, Scalar_ beta, Scalar_* dC, Index_ lddc,
                              magma_queue_t queue) {
	if constexpr(std::is_same_v<Scalar_, float>) {
		return magmablas_ssymm(side, uplo, m, n, alpha, dA, ldda, dB, lddb, beta, dC, lddc, queue);
	}
	else if constexpr(std::is_same_v<Scalar_, double>) {
		return magmablas_dsymm(side, uplo, m, n, alpha, dA, ldda, dB, lddb, beta, dC, lddc, queue);
	}
	else if constexpr(std::is_same_v<Scalar_, cuda::std::complex<float>>) {
		return magma_chemm(side, uplo, m, n, MAGMA_C_MAKE(alpha.real(), alpha.imag()),
		                   reinterpret_cast<magmaFloatComplex const*>(dA), ldda,
		                   reinterpret_cast<magmaFloatComplex const*>(dB), lddb,
		                   MAGMA_C_MAKE(beta.real(), beta.imag()),
		                   reinterpret_cast<magmaFloatComplex*>(dC), lddc, queue);
	}
	else if constexpr(std::is_same_v<Scalar_, cuda::std::complex<double>>) {
		return magma_zhemm(side, uplo, m, n, MAGMA_Z_MAKE(alpha.real(), alpha.imag()),
		                   reinterpret_cast<magmaDoubleComplex const*>(dA), ldda,
		                   reinterpret_cast<magmaDoubleComplex const*>(dB), lddb,
		                   MAGMA_Z_MAKE(beta.real(), beta.imag()),
		                   reinterpret_cast<magmaDoubleComplex*>(dC), lddc, queue);
	}
	else {
		static_assert([] { return false; }(),
		              "Error: magma_hemm for the input scalar type is NOT implemented.");
	}
	return;
}

// MAGMA_LAPACK
template<typename Scalar_, typename Index1_, typename Index2_, typename Index3_, typename Index4_,
         typename Index5_, typename Index6_, typename Index7_ >
static inline Index1_ magma_heevd(magma_vec_t jobz, magma_uplo_t uplo, Index2_ n, Scalar_* A,
                                  Index3_ lda, typename Eigen::NumTraits<Scalar_>::Real* w,
                                  Scalar_* work, Index4_ lwork,
                                  typename Eigen::NumTraits<Scalar_>::Real* rwork, Index5_ lrwork,
                                  Index6_* iwork, Index7_ liwork, Index1_* info) {
	if constexpr(std::is_same_v<Scalar_, float>) {
		return magma_sheevd(jobz, uplo, n, A, lda, w, work, lwork, rwork, lrwork,
		                    reinterpret_cast<magma_int_t*>(iwork), liwork, info);
	}
	else if constexpr(std::is_same_v<Scalar_, double>) {
		return magma_dheevd(jobz, uplo, n, A, lda, w, work, lwork, rwork, lrwork,
		                    reinterpret_cast<magma_int_t*>(iwork), liwork, info);
	}
	else if constexpr(std::is_convertible_v<Scalar_, std::complex<float>>) {
		return magma_cheevd(jobz, uplo, n, reinterpret_cast<magmaFloatComplex*>(A), lda, w,
		                    reinterpret_cast<magmaFloatComplex*>(work), lwork, rwork, lrwork,
		                    reinterpret_cast<magma_int_t*>(iwork), liwork, info);
	}
	else if constexpr(std::is_convertible_v<Scalar_, std::complex<double>>) {
		return magma_zheevd(jobz, uplo, n, reinterpret_cast<magmaDoubleComplex*>(A), lda, w,
		                    reinterpret_cast<magmaDoubleComplex*>(work), lwork, rwork, lrwork,
		                    reinterpret_cast<magma_int_t*>(iwork), liwork, info);
	}
	else {
		static_assert([] { return false; }(),
		              "Error: magma_heevd_gpu for the input scalar type is NOT implemented.");
	}
	// To suppress wrong compiler warning "warning: missing return statement at end of non-void function"
	// Never be executed
	return EXIT_SUCCESS;
}

template<typename Scalar_, typename Index1_, typename Index2_, typename Index3_, typename Index4_,
         typename Index5_, typename Index6_, typename Index7_, typename Index8_ >
static inline Index1_ magma_heevd_gpu(magma_vec_t jobz, magma_uplo_t uplo, Index2_ n, Scalar_* dA,
                                      Index3_ ldda, typename Eigen::NumTraits<Scalar_>::Real* w,
                                      Scalar_* wA, Index4_ ldwa, Scalar_* work, Index5_ lwork,
                                      typename Eigen::NumTraits<Scalar_>::Real* rwork,
                                      Index6_ lrwork, Index7_* iwork, Index8_ liwork,
                                      Index1_* info) {
	if constexpr(std::is_same_v<Scalar_, float>) {
		return magma_sheevd_gpu(jobz, uplo, n, dA, ldda, w, wA, ldwa, work, lwork, rwork, lrwork,
		                        reinterpret_cast<magma_int_t*>(iwork), liwork, info);
	}
	else if constexpr(std::is_same_v<Scalar_, double>) {
		return magma_dheevd_gpu(jobz, uplo, n, dA, ldda, w, wA, ldwa, work, lwork, rwork, lrwork,
		                        reinterpret_cast<magma_int_t*>(iwork), liwork, info);
	}
	else if constexpr(std::is_same_v<Scalar_, cuda::std::complex<float>>) {
		return magma_cheevd_gpu(jobz, uplo, n, reinterpret_cast<magmaFloatComplex*>(dA), ldda, w,
		                        reinterpret_cast<magmaFloatComplex*>(wA), ldwa,
		                        reinterpret_cast<magmaFloatComplex*>(work), lwork, rwork, lrwork,
		                        reinterpret_cast<magma_int_t*>(iwork), liwork, info);
	}
	else if constexpr(std::is_same_v<Scalar_, cuda::std::complex<double>>) {
		return magma_zheevd_gpu(jobz, uplo, n, reinterpret_cast<magmaDoubleComplex*>(dA), ldda, w,
		                        reinterpret_cast<magmaDoubleComplex*>(wA), ldwa,
		                        reinterpret_cast<magmaDoubleComplex*>(work), lwork, rwork, lrwork,
		                        reinterpret_cast<magma_int_t*>(iwork), liwork, info);
	}
	else {
		static_assert([] { return false; }(),
		              "Error: magma_heevd_gpu for the input scalar type is NOT implemented.");
	}
	// To suppress wrong compiler warning "warning: missing return statement at end of non-void function"
	// Never be executed
	return EXIT_SUCCESS;
}

template<typename Scalar_, typename Index1_, typename Index2_, typename Index3_, typename Index4_,
         typename Index5_, typename Index6_ >
static inline Index1_ magma_geev(magma_vec_t jobvl, magma_vec_t jobvr, Index2_ n, Scalar_* A,
                                 Index3_ ldda, Scalar_* w, Scalar_* VL, Index4_ ldvl, Scalar_* VR,
                                 Index5_ ldvr, Scalar_* work, Index6_ lwork,
                                 typename Eigen::NumTraits<Scalar_>::Real* rwork, Index1_* info) {
	if constexpr(std::is_same_v<Scalar_, float>) {
		return magma_sgeev(jobvl, jobvr, n, A, ldda, w, VL, ldvl, VR, ldvr, work, lwork, rwork,
		                   info);
	}
	else if constexpr(std::is_same_v<Scalar_, double>) {
		return magma_dgeev(jobvl, jobvr, n, A, ldda, w, VL, ldvl, VR, ldvr, work, lwork, rwork,
		                   info);
	}
	else if constexpr(std::is_convertible_v<Scalar_, std::complex<float> >) {
		return magma_cgeev(jobvl, jobvr, n, reinterpret_cast<magmaFloatComplex*>(A), ldda,
		                   reinterpret_cast<magmaFloatComplex*>(w),
		                   reinterpret_cast<magmaFloatComplex*>(VL), ldvl,
		                   reinterpret_cast<magmaFloatComplex*>(VR), ldvr,
		                   reinterpret_cast<magmaFloatComplex*>(work), lwork, rwork, info);
	}
	else if constexpr(std::is_convertible_v<Scalar_, std::complex<double> >) {
		return magma_zgeev(jobvl, jobvr, n, reinterpret_cast<magmaDoubleComplex*>(A), ldda,
		                   reinterpret_cast<magmaDoubleComplex*>(w),
		                   reinterpret_cast<magmaDoubleComplex*>(VL), ldvl,
		                   reinterpret_cast<magmaDoubleComplex*>(VR), ldvr,
		                   reinterpret_cast<magmaDoubleComplex*>(work), lwork, rwork, info);
	}
	else {
		static_assert([] { return false; }(),
		              "Error: magma_geevd_gpu for the input scalar type is NOT implemented.");
	}
	// To suppress wrong compiler warning "warning: missing return statement at end of non-void function"
	// Never be executed
	return EXIT_SUCCESS;
}