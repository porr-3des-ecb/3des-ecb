cmake_minimum_required (VERSION 3.5)

set (tdes_sequential_sources
	main.cpp
	TDESSequential.cpp
	../TDESInterface.cpp
)

add_executable(tdes_sequential ${tdes_sequential_sources})
