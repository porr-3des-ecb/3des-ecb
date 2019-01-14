cmake_minimum_required (VERSION 3.10)

find_package(CUDA REQUIRED)
enable_language(CUDA)

set (tdes_cuda_sources
	main.cu
	TDESCuda.cu
	../TDESInterface.cpp
	../common/des_helpers.cpp
)

add_executable(tdes_cuda ${tdes_cuda_sources})
# target_compile_options(tdes_cuda PRIVATE ${OpenMP_CXX_FLAGS})
# target_link_libraries(tdes_cuda PRIVATE ${OpenMP_CXX_FLAGS})
