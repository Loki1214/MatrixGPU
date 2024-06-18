#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <unsupported/Eigen/MPRealSupport>
#include <random>

int main(int argc, char** argv) {
	if(argc < 2) {
		std::cerr << "Usage: 0.(This) 1.(dim)\n";
		std::cerr << "argc = " << argc << std::endl;
		std::exit(EXIT_FAILURE);
	}
	int const dim = std::atoi(argv[1]);

	// set precision to 256 bits (double has only 53 bits)
	mpfr::mpreal::set_default_prec(256);
	using RealScalar = mpfr::mpreal;
	using Scalar = std::complex<RealScalar>;

	std::random_device                   seed_gen;
	std::mt19937                         engine(seed_gen());
	std::normal_distribution<double> dist(0.0, 1.0);

	Eigen::MatrixX<Scalar> mat = Eigen::MatrixX<Scalar>::NullaryExpr(dim, dim,
		[&](){ return Scalar(dist(engine), dist(engine)); }
	);
	mat /= mat.norm();

	Eigen::ComplexEigenSolver<decltype(mat)> solver(mat, true);
	std::cout << solver.eigenvalues() << std::endl;
	std::cout << solver.eigenvectors() << std::endl;

	return EXIT_SUCCESS;
}