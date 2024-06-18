cmake_minimum_required(VERSION 3.14)
project(MultiPrecisionComplexEigenSolver CXX)
set(CMAKE_CXX_STANDARD 17)

find_package(Eigen3 REQUIRED)
find_package(OpenMP REQUIRED)

set(modules
MultiPrecisionComplexEigenSolver
)
foreach(name IN LISTS modules)
	add_executable(${name} ${name}.cpp)
	target_compile_options(${name} PRIVATE -Wall -Wextra -Wfatal-errors)
	target_include_directories(${name} SYSTEM PUBLIC "./mpreal-master")
	target_link_libraries(${name}
		Eigen3::Eigen
		OpenMP::OpenMP_CXX
		-lmpfr
	)
endforeach(name)
