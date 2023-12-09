#pragma once

#include "macros.hpp"
#include "MatrixGPU.cuh"
#include "magma_overloads.cuh"

namespace GPU {
	template<class Matrix_>
	class ComplexEigenSolver;

	template<typename _Scalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
	class ComplexEigenSolver< Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols> > {
		public:
			using MatrixCPU  = Eigen::Matrix<_Scalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>;
			using VectorCPU  = Eigen::VectorX<_Scalar>;
			using Index      = typename MatrixCPU::Index;
			using ScalarCPU  = typename MatrixCPU::Scalar;
			using RealScalar = typename MatrixCPU::RealScalar;

		private:
			MatrixCPU   m_VL;
			MatrixCPU   m_VR;
			VectorCPU   m_eigvals;
			magma_int_t m_info;

		public:
			ComplexEigenSolver() = default;
			ComplexEigenSolver(MatrixCPU const& hmat) { this->compute(hmat); }
			MatrixCPU           eigenvectors() const { return m_VR; }
			MatrixCPU           leftEigenvectors() const { return m_VL; }
			VectorCPU const&    eigenvalues() const { return m_eigvals; }
			ComplexEigenSolver& compute(MatrixCPU const& hmat, bool computeLeftEigenvectors = true,
			                            bool computeRightEigenvectors = true) {
				DEBUG(std::cerr << "# " << __func__ << std::endl);
				magma_vec_t jobvl = (computeLeftEigenvectors ? MagmaVec : MagmaNoVec);
				magma_vec_t jobvr = (computeRightEigenvectors ? MagmaVec : MagmaNoVec);
				if(computeLeftEigenvectors) m_VL.resize(hmat.rows(), hmat.cols());
				if(computeRightEigenvectors) m_VR.resize(hmat.rows(), hmat.cols());
				Eigen::VectorX<ScalarCPU>  work(1);
				Eigen::VectorX<RealScalar> rwork(2 * hmat.rows());

				MatrixCPU hmatCopy(hmat);
				m_eigvals.resize(hmat.rows());
				magma_geev_gpu(jobvl, jobvr, hmatCopy.rows(), hmatCopy.data(), hmatCopy.rows(),
				               m_eigvals.data(), m_VL.data(), m_VL.rows(), m_VR.data(), m_VR.rows(),
				               work.data(), -1, rwork.data(), &m_info);
				DEBUG(std::cerr << "#         hmatCopy.rows() = " << hmatCopy.rows() << std::endl);
				DEBUG(std::cerr << "#         hmatCopy.cols() = " << hmatCopy.cols() << std::endl);
				DEBUG(std::cerr << "# int(std::real(work[0])) = " << int(std::real(work[0]))
				                << std::endl);
				work.resize(int(std::real(work[0])));
				DEBUG(std::cerr << "#             work.size() = " << work.size() << std::endl);
				magma_geev_gpu(jobvl, jobvr, hmatCopy.rows(), hmatCopy.data(), hmatCopy.rows(),
				               m_eigvals.data(), m_VL.data(), m_VL.rows(), m_VR.data(), m_VR.rows(),
				               work.data(), work.size(), rwork.data(), &m_info);
				DEBUG(std::cerr << "# info = " << m_info << "\n" << std::endl);
				return *this;
			}
	};

}  // namespace GPU