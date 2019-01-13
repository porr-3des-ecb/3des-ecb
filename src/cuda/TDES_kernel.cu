#include <cuda_runtime.h>
//#include <helper_functions.h>
//#include <helper_cuda.h>
#include "device_launch_parameters.h"
#include "TDESCuda.cuh"
#include <string>
#include <sstream>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <cstring>

__global__
void encodeK(char* in, char* out,unsigned int size)
{
    unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
	if(index>=size)
	{
		return;
	}
	
	// Parse hex block into 64-bit
	uint64_t block = *((uint64_t*)(in+16*index));//std::stoull(message.substr(16 * i, 16), 0, 16);
	// Encode with k1, decode with k2, encode with k3
	uint64_t blockPass1 = processBlock(block, 0, false);
	uint64_t blockPass2 = processBlock(blockPass1, 1, true);
	uint64_t blockPass3 = processBlock(blockPass2, 2, false);

	memcpy(out+16*index, &blockPass2,16);
}

__global__
void decodeK(char* in, char* out,unsigned int size)
{
	unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
	if(index>=size)
	{
		return;
	}
	
	// Parse hex block into 64-bit
	uint64_t block = *((uint64_t*)(in+16*index));//std::stoull(message.substr(16 * i, 16), 0, 16);
	// Decode with k3, encode with k2, decode with k1
	uint64_t blockPass1 = processBlock(block, 2, true);
	uint64_t blockPass2 = processBlock(blockPass1, 1, false);
	uint64_t blockPass3 = processBlock(blockPass2, 0, true);

	memcpy(out+16*index,&blockPass3,16);
}