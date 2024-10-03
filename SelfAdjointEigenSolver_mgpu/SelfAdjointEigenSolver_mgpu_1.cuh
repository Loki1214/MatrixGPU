
#pragma once

#include "macros.hpp"
#include "MatrixGPU.cuh"
#include "magma_overloads.cuh"
#include <vector>

namespace GPU {
	template<class Matrix_>
	class SelfAdjointEigenSolver_mgpu;

	template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
	class SelfAdjointEigenSolver_mgpu<
	    Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> > {
		public:
			using MatrixCPU  = Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>;
			using Index      = typename MatrixCPU::Index;
			using ScalarCPU  = typename MatrixCPU::Scalar;
			using RealScalar = typename MatrixCPU::RealScalar;
			using VectorCPU  = Eigen::VectorX<RealScalar>;

		private:
			MatrixCPU   m_eigvecs;
			VectorCPU   m_eigvals;
			magma_int_t m_info;

		public:
			SelfAdjointEigenSolver_mgpu() = default;
			SelfAdjointEigenSolver_mgpu(int const ngpu, MatrixCPU const& hmat,
			                            Eigen::DecompositionOptions option
			                            = Eigen::ComputeEigenvectors) {
				this->compute(ngpu, hmat, option);
			}
			MatrixCPU const&             eigenvectors() const { return m_eigvecs; }
			VectorCPU const&             eigenvalues() const { return m_eigvals; }
			SelfAdjointEigenSolver_mgpu& compute(int const ngpu, MatrixCPU const& hmat,
			                                     Eigen::DecompositionOptions option
			                                     = Eigen::ComputeEigenvectors) {
				DEBUG(std::cerr << "# " << __func__ << std::endl);

				magma_vec_t  jobz = (option == Eigen::ComputeEigenvectors ? MagmaVec : MagmaNoVec);
				magma_uplo_t uplo = MagmaLower;
				using ScalarGPU   = magmaDoubleComplex;
				// std::vector<ScalarGPU>   work(1);
				// std::vector<RealScalar>  rwork(1);
				// std::vector<magma_int_t> iwork(1);

				// m_eigvecs = hmat;
				m_eigvals.resize(hmat.rows());
				magma_int_t const dim = hmat.rows();

				std::vector<magmaDoubleComplex> mat(hmat.rows() * hmat.cols());
				double                          norm = 0;
#pragma omp parallel for schedule(dynamic, 100) reduction(+ : norm)
				for(auto j = 0; j < hmat.rows(); ++j) {
					mat.at(j + dim * j) = MAGMA_Z_MAKE(real(hmat(j, j)), 0);
					norm += real(mat.at(j + dim * j) * conj(mat.at(j + dim * j)));
					for(auto k = 0; k < j; ++k) {
						mat.at(j + dim * k) = MAGMA_Z_MAKE(real(hmat(j, k)), imag(hmat(j, k)));
						mat.at(k + dim * j) = conj(mat.at(j + hmat.rows() * k));
						norm += 2 * real(mat.at(j + dim * k) * conj(mat.at(j + dim * k)));
					}
				}
				norm = std::sqrt(norm);
#pragma omp parallel for
				for(Eigen::Index j = 0; j < mat.size(); ++j) { mat.at(j) /= norm; }
				std::cout << "## Prepared a matrix on CPU." << std::endl;

				magma_int_t                     info = 0;
				std::vector<double>             eigVals(dim);
				std::vector<magmaDoubleComplex> work(1);
				std::vector<double>             rwork(1);
				std::vector<magma_int_t>        iwork(1);
				magma_zheevd_m(ngpu, MagmaVec, MagmaLower, dim,
				               reinterpret_cast<magmaDoubleComplex*>(mat.data()), dim,
				               eigVals.data(), work.data(), -1, rwork.data(), -1, iwork.data(), -1,
				               &info);
				magma_int_t const lwork  = magma_int_t(real(work[0]));
				magma_int_t const lrwork = magma_int_t(rwork[0]);
				magma_int_t const liwork = iwork[0];
				std::cout << "#\t  lwork = " << lwork << "\n"
				          << "#\t lrwork = " << lrwork << "\n"
				          << "#\t liwork = " << liwork << std::endl;
				work.resize(lwork);
				rwork.resize(lrwork);
				iwork.resize(liwork);
				magma_zheevd_m(ngpu, MagmaVec, MagmaLower, dim,
				               reinterpret_cast<magmaDoubleComplex*>(mat.data()), dim,
				               eigVals.data(), work.data(), lwork, rwork.data(), lrwork,
				               iwork.data(), liwork, &info);
				std::cout << "#\t info = " << info << std::endl;

				// std::vector<magmaDoubleComplex> work(1);
				// std::vector<double>             rwork(1);
				// std::vector<magma_int_t>        iwork(1);
				// magma_heevd_m(ngpu, jobz, uplo, dim, mat.data(),
				//               dim, m_eigvals.data(), work.data(), -1, rwork.data(), -1,
				//               iwork.data(), -1, &m_info);
				// DEBUG(std::cerr << "#           dim = " << dim << std::endl);
				// DEBUG(std::cerr << "#           m_eigvecs.rows() = " << m_eigvecs.rows()
				//                 << std::endl);
				// DEBUG(std::cerr << "#           m_eigvecs.cols() = " << m_eigvecs.cols()
				//                 << std::endl);
				// DEBUG(std::cerr << "# magma_int_t(real(work[0])) = " << magma_int_t(real(work[0]))
				//                 << std::endl);
				// DEBUG(std::cerr << "#      magma_int_t(rwork[0]) = " << magma_int_t(rwork[0])
				//                 << std::endl);
				// DEBUG(std::cerr << "#                 rwork[0])  = " << rwork[0] << std::endl);
				// DEBUG(std::cerr << "#                   iwork[0] = " << iwork[0] << std::endl);
				// work.resize(magma_int_t(real(work[0])));
				// rwork.resize(magma_int_t(rwork[0]));
				// iwork.resize(iwork[0]);
				// DEBUG(std::cerr << "#        work.size() = " << work.size() << std::endl);
				// DEBUG(std::cerr << "#       rwork.size() = " << rwork.size() << std::endl);
				// DEBUG(std::cerr << "#       iwork.size() = " << iwork.size() << std::endl);
				// magma_heevd_m(ngpu, jobz, uplo, dim, mat.data(),
				//               dim, m_eigvals.data(), work.data(), magma_int_t(work.size()),
				//               rwork.data(), magma_int_t(rwork.size()), iwork.data(),
				//               magma_int_t(iwork.size()), &m_info);
				// DEBUG(std::cerr << "# info = " << m_info << "\n" << std::endl);
				return *this;
			}
	};
}  // namespace GPU