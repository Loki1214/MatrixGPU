cmake_minimum_required(VERSION 3.14)
set(CMAKE_CXX_STANDARD 17)

include_directories(SYSTEM "${PROJECT_SOURCE_DIR}")
include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

# set(CMAKE_CXX_STANDARD 17)
# if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
# 	set(CMAKE_CUDA_ARCHITECTURES 70)
# endif()
# enable_language(CUDA)
# # set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
# # set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
# #   set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --relocatable-device-code=true")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -Xcompiler -rdynamic")
# find_package(CUDAToolkit REQUIRED)

# find_package(PkgConfig)
# pkg_check_modules(MAGMA REQUIRED IMPORTED_TARGET magma)

find_package(Eigen3 REQUIRED)

find_package(BLAS REQUIRED)
find_package(LAPACK REQUIRED)
message(${LAPACK_LIBRARIES})
string(FIND "${LAPACK_LIBRARIES}" "libiomp" USE_libiomp)
if("${USE_libiomp}" EQUAL "-1")
	set(USE_libiomp FALSE)
	message("libiomp is not found.")
else()
	set(USE_libiomp TRUE)
	message("libiomp is found. Using libiomp.")
	include_directories(SYSTEM "$ENV{MKLROOT}/include")
	message("MKLROOT/include = $ENV{MKLROOT}/include")
endif()

# For OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
	# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")
endif()
if(USE_libiomp)
	set_property(TARGET OpenMP::OpenMP_CXX
		PROPERTY INTERFACE_LINK_LIBRARIES "")
endif()



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
		LAPACK::LAPACK
		BLAS::BLAS
		PkgConfig::MAGMA
	)

	set(name "${prefix}${obsName}_c")
	add_executable(${name} ${obsName}.cpp)
	set_target_properties(${name} PROPERTIES OUTPUT_NAME "${obsName}_c")
	target_compile_options(${name} PRIVATE -Wall -Wextra -Wfatal-errors -DFLOAT)
	target_link_libraries(${name}
		Eigen3::Eigen
		OpenMP::OpenMP_CXX
		LAPACK::LAPACK
		BLAS::BLAS
		PkgConfig::MAGMA
	)
endforeach(obsName)

enable_language(CUDA)
set(prefix "benchmark_")
set(modules
	SelfAdjointEigenSolverGPU
	ComplexEigenSolverGPU
)
foreach(obsName IN LISTS modules)
	set(name "${prefix}${obsName}")
	add_executable(${name} ${obsName}.cu)
	set_target_properties(${name} PROPERTIES OUTPUT_NAME "${obsName}")
	target_compile_options(${name} PRIVATE -Wall -Wextra -Wfatal-errors)
	target_link_libraries(${name}
		Eigen3::Eigen
		OpenMP::OpenMP_CXX
		LAPACK::LAPACK
		BLAS::BLAS
		PkgConfig::MAGMA
	)

	set(name "${prefix}${obsName}_c")
	add_executable(${name} ${obsName}.cu)
	set_target_properties(${name} PROPERTIES OUTPUT_NAME "${obsName}_c")
	target_compile_options(${name} PRIVATE -Wall -Wextra -Wfatal-errors -DFLOAT)
	target_link_libraries(${name}
		Eigen3::Eigen
		OpenMP::OpenMP_CXX
		LAPACK::LAPACK
		BLAS::BLAS
		PkgConfig::MAGMA
	)
endforeach(obsName)
