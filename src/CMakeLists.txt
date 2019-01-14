cmake_minimum_required (VERSION 3.10)

project (TDES_ECB
	VERSION 0.0.1
	DESCRIPTION "Comparison of different parallel/concurrent implementations of a 3DES-ECB cipher"
	LANGUAGES CXX
)

# Set standard to C++17
if(NOT CMAKE_CXX_STANDARD)
	set(CMAKE_CXX_STANDARD 17)
endif()

# Add warnings for GCC/Clang
# if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
# 	add_compile_options(-Wall -Wextra -Wpedantic)
# endif()

add_subdirectory(sequential)
add_subdirectory(openmp)
add_subdirectory(cuda)
