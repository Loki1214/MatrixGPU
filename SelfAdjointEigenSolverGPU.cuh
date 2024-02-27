
#pragma once

#include "macros.hpp"
#include "MatrixGPU.cuh"
#include "magma_overloads.cuh"

namespace GPU {
	template<class Matrix_>
	class SelfAdjointEigenSolver;

	template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
	class SelfAdjointEigenSolver<
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
			SelfAdjointEigenSolver() = default;
			SelfAdjointEigenSolver(MatrixCPU const& hmat) { this->compute(hmat); }
			MatrixCPU const&        eigenvectors() const { return m_eigvecs; }
			VectorCPU const&        eigenvalues() const { return m_eigvals; }
			SelfAdjointEigenSolver& compute(MatrixCPU const& hmat,
			                                bool             computeEigenvectors = true) {
				DEBUG(std::cerr << "# " << __func__ << std::endl);
				magma_vec_t                jobz = (computeEigenvectors ? MagmaVec : MagmaNoVec);
				magma_uplo_t               uplo = MagmaLower;
				Eigen::VectorX<ScalarCPU>  work(1);
				Eigen::VectorX<RealScalar> rwork(1);
				Eigen::VectorXi            iwork(1);

				m_eigvecs = hmat;
				m_eigvals.resize(hmat.rows());
				magma_heevd(jobz, uplo, m_eigvecs.rows(), m_eigvecs.data(), m_eigvecs.rows(),
				            m_eigvals.data(), work.data(), -1, rwork.data(), -1, iwork.data(), -1,
				            &m_info);
				DEBUG(std::cerr << "#   m_eigvecs.rows() = " << m_eigvecs.rows() << std::endl);
				DEBUG(std::cerr << "#   m_eigvecs.cols() = " << m_eigvecs.cols() << std::endl);
				DEBUG(std::cerr << "# int(real(work[0])) = " << int(real(work[0])) << std::endl);
				DEBUG(std::cerr << "#     int(rwork[0])) = " << int(rwork[0]) << std::endl);
				DEBUG(std::cerr << "#           iwork[0] = " << iwork[0] << std::endl);
				work.resize(int(real(work[0])));
				rwork.resize(int(rwork[0]));
				iwork.resize(iwork[0]);
				DEBUG(std::cerr << "#        work.size() = " << work.size() << std::endl);
				DEBUG(std::cerr << "#       rwork.size() = " << rwork.size() << std::endl);
				DEBUG(std::cerr << "#       iwork.size() = " << iwork.size() << std::endl);
				magma_heevd(jobz, uplo, m_eigvecs.rows(), m_eigvecs.data(), m_eigvecs.rows(),
				            m_eigvals.data(), work.data(), work.size(), rwork.data(), rwork.size(),
				            iwork.data(), iwork.size(), &m_info);
				DEBUG(std::cerr << "# info = " << m_info << "\n" << std::endl);
				return *this;
			}
	};

	template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
	class SelfAdjointEigenSolver<
	    MatrixGPU<Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>> > {
		public:
			using MatrixCPU  = Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>;
			using Index      = typename MatrixCPU::Index;
			using ScalarCPU  = typename MatrixCPU::Scalar;
			using RealScalar = typename MatrixCPU::RealScalar;
			using VectorCPU  = Eigen::VectorX<RealScalar>;

			using MatrixGPU = MatrixGPU<MatrixCPU>;
			using ScalarGPU = typename MatrixGPU::Scalar;

		private:
			MatrixGPU   m_eigvecs;
			VectorCPU   m_eigvals;
			magma_int_t m_info;

		public:
			SelfAdjointEigenSolver() = default;
			SelfAdjointEigenSolver(MatrixGPU const& dmat) { this->compute(dmat); }
			MatrixGPU const& eigenvectorsGPU() const { return m_eigvecs; }
			MatrixCPU        eigenvectors() const {
                MatrixCPU res(m_eigvecs.rows(), m_eigvecs.cols());
                m_eigvecs.copyTo(res);
                return res;
			}
			VectorCPU const&        eigenvalues() const { return m_eigvals; }
			SelfAdjointEigenSolver& compute(MatrixGPU const& dmat,
			                                bool             computeEigenvectors = true) {
				DEBUG(std::cerr << "# " << __func__ << std::endl);
				magma_vec_t                jobz = (computeEigenvectors ? MagmaVec : MagmaNoVec);
				magma_uplo_t               uplo = MagmaLower;
				Eigen::MatrixX<ScalarGPU>  wA(dmat.rows(), dmat.cols());
				Eigen::VectorX<ScalarGPU>  work(1);
				Eigen::VectorX<RealScalar> rwork(1);
				Eigen::VectorXi            iwork(1);

				m_eigvecs = dmat;
				m_eigvals.resize(dmat.rows());
				magma_heevd_gpu(jobz, uplo, m_eigvecs.rows(), m_eigvecs.data(), m_eigvecs.LD(),
				                m_eigvals.data(), wA.data(), wA.rows(), work.data(), -1,
				                rwork.data(), -1, iwork.data(), -1, &m_info);
				DEBUG(std::cerr << "#   m_eigvecs.rows() = " << m_eigvecs.rows() << std::endl);
				DEBUG(std::cerr << "#   m_eigvecs.cols() = " << m_eigvecs.cols() << std::endl);
				DEBUG(std::cerr << "# int(real(work[0])) = " << int(real(work[0])) << std::endl);
				DEBUG(std::cerr << "#     int(rwork[0])) = " << int(rwork[0]) << std::endl);
				DEBUG(std::cerr << "#           iwork[0] = " << iwork[0] << std::endl);
				work.resize(int(real(work[0])));
				rwork.resize(int(rwork[0]));
				iwork.resize(iwork[0]);
				DEBUG(std::cerr << "#        work.size() = " << work.size() << std::endl);
				DEBUG(std::cerr << "#       rwork.size() = " << rwork.size() << std::endl);
				DEBUG(std::cerr << "#       iwork.size() = " << iwork.size() << std::endl);
				magma_heevd_gpu(jobz, uplo, m_eigvecs.rows(), m_eigvecs.data(), m_eigvecs.LD(),
				                m_eigvals.data(), wA.data(), wA.rows(), work.data(), work.size(),
				                rwork.data(), rwork.size(), iwork.data(), iwork.size(), &m_info);
				DEBUG(std::cerr << "# info = " << m_info << "\n" << std::endl);
				return *this;
			}
	};

}  // namespace GPU