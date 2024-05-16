enable_language(CXX)
set(prefix "benchmark_")
set(modules
	SelfAdjointEigenSolver
	ComplexEigenSolver
)
foreach(obsName IN LISTS modules)
	set(name "${prefix}${obsName}")
	add_executable(${name} ${obsName}.cpp)
	set_target_properties(${name} PROPERTIES OUTPUT_NAME "${obsName}")
	target_compile_options(${name} PRIVATE -Wall -Wextra -Wfatal-errors)
	target_link_libraries(${name}
		Eigen3::Eigen
		OpenMP::OpenMP_CXX
		MKL::MKL
	)

	set(name "${prefix}${obsName}_c")
	add_executable(${name} ${obsName}.cpp)
	set_target_properties(${name} PROPERTIES OUTPUT_NAME "${obsName}_c")
	target_compile_options(${name} PRIVATE -Wall -Wextra -Wfatal-errors -DFLOAT)
	target_link_libraries(${name}
		Eigen3::Eigen
		OpenMP::OpenMP_CXX
		MKL::MKL
	)
endforeach(obsName)

enable_language(CUDA)
set(prefix "benchmark_")
set(modules
	SelfAdjointEigenSolverGPU
	SelfAdjointEigenSolver_mgpu
	ComplexEigenSolverGPU
)
foreach(obsName IN LISTS modules)
	set(name "${prefix}${obsName}")
	add_executable(${name} ${obsName}.cu)
	set_target_properties(${name} PROPERTIES OUTPUT_NAME "${obsName}")
	target_compile_options(${name} PRIVATE -Wall -Wextra -Wfatal-errors)
	target_link_libraries(${name}
		Eigen3::Eigen
		MKL::MKL
		OpenMP::OpenMP_CXX
		PkgConfig::MAGMA
	)

	set(name "${prefix}${obsName}_c")
	add_executable(${name} ${obsName}.cu)
	set_target_properties(${name} PROPERTIES OUTPUT_NAME "${obsName}_c")
	target_compile_options(${name} PRIVATE -Wall -Wextra -Wfatal-errors -DFLOAT)
	target_link_libraries(${name}
		Eigen3::Eigen
		MKL::MKL
		OpenMP::OpenMP_CXX
		PkgConfig::MAGMA
	)
endforeach(obsName)
