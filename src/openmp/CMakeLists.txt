cmake_minimum_required (VERSION 3.10)

find_package(OpenMP REQUIRED)

set (tdes_openmp_sources
	main.cpp
	TDESOpenMP.cpp
	../TDESInterface.cpp
	../common/des_helpers.cpp
)

add_executable(tdes_openmp ${tdes_openmp_sources})
target_compile_options(tdes_openmp PRIVATE ${OpenMP_CXX_FLAGS} -Wall -Wextra -Wpedantic)
target_link_libraries(tdes_openmp PRIVATE ${OpenMP_CXX_FLAGS})
