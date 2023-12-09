cmake_minimum_required(VERSION 3.14)
project(MatrixGPU CXX)
set(CMAKE_CXX_STANDARD 17)
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
	set(CMAKE_CUDA_ARCHITECTURES 70)
endif()
enable_language(CUDA)
# set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
# set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
#   set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --relocatable-device-code=true")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcudafe --diag_suppress=esa_on_defaulted_function_ignored")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -lineinfo -Xcompiler -rdynamic")
find_package(CUDAToolkit REQUIRED)

find_package(PkgConfig)
pkg_check_modules(MAGMA REQUIRED IMPORTED_TARGET magma)
# include_directories(SYSTEM ${MAGMA_INCLUDE_DIRS})
# string(REPLACE "=" "" MAGMA_INCLUDE_DIRS "${MAGMA_INCLUDE_DIRS}")
# message("MAGMA_INCLUDE_DIRS = ${MAGMA_INCLUDE_DIRS}")
# message("MAGMA_LIBRARY_DIRS = ${MAGMA_LIBRARY_DIRS}")
# message("   MAGMA_LIBRARIES = ${MAGMA_LIBRARIES}")

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

# find_package(MAGMA REQUIRED)

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
	set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -fopenmp")
endif()
if(USE_libiomp)
	set_property(TARGET OpenMP::OpenMP_CXX
		PROPERTY INTERFACE_LINK_LIBRARIES "")
endif()


include_directories("${PROJECT_SOURCE_DIR}")
set(modules
	MatrixGPU
	SelfAdjointEigenSolverGPU
	ComplexEigenSolverGPU
)

foreach(name IN LISTS modules)
	add_executable(${name}_test ${name}_test.cu)
	target_compile_options(${name}_test PRIVATE -Wall -Wextra -Wfatal-errors)
	target_link_libraries(${name}_test
		Catch2::Catch2WithMain
		Eigen3::Eigen
		OpenMP::OpenMP_CXX
		LAPACK::LAPACK
		BLAS::BLAS
		PkgConfig::MAGMA
	)
	add_test(${name}_test ${name}_test)
endforeach(name)
