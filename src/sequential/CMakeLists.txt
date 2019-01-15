cmake_minimum_required (VERSION 3.10)

set (tdes_sequential_sources
	main.cpp
	TDESSequential.cpp
	../TDESInterface.cpp
	../common/des_helpers.cpp
)

add_executable(tdes_sequential ${tdes_sequential_sources})
target_compile_options(tdes_sequential PRIVATE -Wall -Wextra -Wpedantic)
