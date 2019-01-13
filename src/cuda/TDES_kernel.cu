#include <cuda_runtime.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <string>
#include <sstream>
#include <cstdlib>

__global__
void encode(char* in, char* out,unsigned int size)
{
    uint index = blockIdx.x*blockDim.x + threadIdx.x;
	if(index>=size)
	{
		return;
	}
	
	// Parse hex block into 64-bit
	uint64_t block = *((uint64_t*)(in+16*index));//std::stoull(message.substr(16 * i, 16), 0, 16);
	// Encode with k1, decode with k2, encode with k3
	uint64_t blockPass1 = this->processBlock(block, 0, false);
	uint64_t blockPass2 = this->processBlock(blockPass1, 1, true);
	uint64_t blockPass3 = this->processBlock(blockPass2, 2, false);

	// Return as hex string
	std::stringstream hexString;
	hexString << std::hex << std::setfill('0') << std::setw(16) << blockPass3;
	cudaMemcpy(out+32*i,hexString.str().z_str(),32,cudaMemcpyDeviceToDevice);
}

__global__
void decode(char* in, char* out,unsigned int size)
{
	uint index = blockIdx.x*blockDim.x + threadIdx.x;
	if(index>=size)
	{
		return;
	}
	
	// Parse hex block into 64-bit
	uint64_t block = *((uint64_t*)(in+16*index));//std::stoull(message.substr(16 * i, 16), 0, 16);
	// Decode with k3, encode with k2, decode with k1
	uint64_t blockPass1 = this->processBlock(block, 2, true);
	uint64_t blockPass2 = this->processBlock(blockPass1, 1, false);
	uint64_t blockPass3 = this->processBlock(blockPass2, 0, true);

	// Return as hex string
	std::stringstream hexString;
	hexString << std::hex << std::setfill('0') << std::setw(16) << blockPass3;
	cudaMemcpy(out+32*i,hexString.str().z_str(),32,cudaMemcpyDeviceToDevice);
}