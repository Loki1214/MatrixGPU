cmake_minimum_required(VERSION 3.14)
project(HilbertSpace CXX)
set(CMAKE_CXX_STANDARD 17)

include(FetchContent)
find_package(Catch2 QUIET)
if(NOT Catch2_FOUND)
	FetchContent_Declare(catch2
		GIT_REPOSITORY https://github.com/catchorg/Catch2.git
		GIT_TAG v3.1.0
	)
	FetchContent_MakeAvailable(Catch2)
endif()
enable_testing()

find_package(Eigen3 REQUIRED)

# For BLAS and LAPACK
find_package(MKL REQUIRED)
find_package(TBB REQUIRED)
find_package(OpenMP REQUIRED)

find_package(PkgConfig)
pkg_check_modules(MAGMA REQUIRED IMPORTED_TARGET magma)


include(CheckLanguage)
check_language(CUDA)
if(CMAKE_CUDA_COMPILER)
enable_language(CUDA)
  execute_process(COMMAND "${CMAKE_SOURCE_DIR}/get_sm.sh" OUTPUT_VARIABLE DETECTED_ARCHITECTURES)
  string(REGEX REPLACE "\n" "" DETECTED_ARCHITECTURES "${DETECTED_ARCHITECTURES}")
  set(CMAKE_CUDA_ARCHITECTURES ${DETECTED_ARCHITECTURES} CACHE STRING "")
  string(REGEX REPLACE "[ \t\r\n]+" "" CMAKE_CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}" )
  message("CMAKE_CUDA_COMPILER_ID   = ${CMAKE_CUDA_COMPILER_ID}")
  message("CMAKE_CUDA_ARCHITECTURES = ${CMAKE_CUDA_ARCHITECTURES}")
  set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -Xcompiler -rdynamic")
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Wall -Wextra -Wfatal-errors")
  if(OpenMP_FOUND)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")
  endif()

  find_package(CUDAToolkit REQUIRED)
else()
  message(STATUS "No CUDA support")
endif()


include_directories("${PROJECT_SOURCE_DIR}")
set(modules
	MatrixGPU
	SelfAdjointEigenSolverGPU
	ComplexEigenSolverGPU
	SelfAdjointEigenSolver_mgpu
)
foreach(name IN LISTS modules)
	add_executable(${name}_test ${name}_test.cu)
	target_compile_options(${name}_test PRIVATE -Wall -Wextra -Wfatal-errors)
	target_link_libraries(${name}_test
		Catch2::Catch2WithMain
		Eigen3::Eigen
		MKL::MKL
		OpenMP::OpenMP_CXX
		PkgConfig::MAGMA
	)
	add_test(${name}_test compute-sanitizer ${name}_test)

	# add_executable(${name}_c_test ${name}_test.cu)
	# target_compile_options(${name}_c_test PRIVATE -Wall -Wextra -Wfatal-errors -DFLOAT)
	# target_link_libraries(${name}_c_test
	# 	Catch2::Catch2WithMain
	# 	Eigen3::Eigen
	# 	MKL::MKL
	# 	OpenMP::OpenMP_CXX
	# 	PkgConfig::MAGMA
	# 	CUDA::curand
	# )
	# add_test(${name}_c_test compute-sanitizer ${name}_c_test)
endforeach(name)

subdirs(benchmark)