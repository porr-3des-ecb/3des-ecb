#include "TDESCuda.cuh"
#include "../common/des_defines.hpp"
#include "../common/des_helpers.hpp"

#include "TDES_kernel.cuh"
#include "device_launch_parameters.h"

#include <iomanip>
#include <iostream>
#include <string>
#include <sstream>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdlib>
#include <cstring>
//#include <helper_functions.h>
//#include <helper_cuda.h>

TDESCuda::~TDESCuda() {}

__constant__ uint64_t ckeys[3*16];

__device__
uint64_t permute(const uint64_t in, const int inSize, const int outSize, const uint8_t * table) {
	uint64_t out = 0;

	for (int i = 0; i < outSize; ++i) {
		out = out << 1;
		out += (in >> (inSize - table[i])) & 0x01;
	}

	return out;
}

void TDESCuda::prepareKeys() {
	for (int k = 0; k < 3; ++k) {
		uint64_t key = this->keys[k];

		// Initial key permutation
		uint64_t permutedKey = TDES::permute(key, 64, 56, PERMUTATION_TABLE_PC1);

		// Left/right half-key rotations
		uint64_t previousKey = permutedKey;
		uint64_t pKeys[16];
		for (int i = 0; i < 16; ++i) {
			// Split the key in 2
			uint64_t lKey = (previousKey >> 28) & 0x0fffffff;
			uint64_t rKey = previousKey & 0x0fffffff;

			// Left shift and rotate the key parts the correct number of times
			int shifts = (i == 0 || i == 1 || i == 8 || i == 15) ? 1 : 2;
			for (int j = 0; j < shifts; ++j) {
				lKey = TDES::lshift(lKey, 28);
				rKey = TDES::lshift(rKey, 28);
			}

			// Save the key
			previousKey = (lKey << 28) + rKey;
			pKeys[i] = previousKey;
		}

		// Final key permutation
		for (int i = 0; i < 16; ++i) {
			this->pKeys[k][i] = TDES::permute(pKeys[i], 56, 48, PERMUTATION_TABLE_PC2);
		}
	}
}

__device__
uint64_t processBlock(uint64_t block, int key, bool decode) {
	// Initial permutation
	uint64_t permutedBlock = permute(block, 64, 64, PERMUTATION_TABLE_IP);

	// Encoding (16 passes)
	uint64_t previousBlock = permutedBlock;
	for (int i = 0; i < 16; ++i) {
		// Split previous block in 2
		uint64_t previousLBlock = (previousBlock >> 32) & 0xffffffff;
		uint64_t previousRBlock = previousBlock & 0xffffffff;

		// Extend it (32b -> 48b)
		uint64_t extendedBlock = permute(previousRBlock, 32, 48, SELECTION_TABLE_E);

		// XOR the extended block with a prepared key
		uint64_t pKey = ckeys[key*16+i];
		if (decode) {
			pKey = ckeys[key * 16 + 15 - i];
		}
		uint64_t xoredBlock = pKey ^ extendedBlock;

		// Do the "selection-boxes" magic
		uint64_t boxSelectedBlock = 0;
		for (int j = 0; j < 8; ++j) {
			// Take 6 bits, starting with the highest
			uint8_t subBlock = (xoredBlock >> (6 * (7 - j))) & 0x3f;
			// Row is the highest (6th) and the first bit
			uint8_t row = 2 * (subBlock >> 5) + (subBlock & 0x01);
			// Column is the middle 4 bits
			uint8_t column = (subBlock >> 1) & 0x0f;

			boxSelectedBlock = boxSelectedBlock << 4;
			boxSelectedBlock += SELECTION_BOX[j][16 * row + column];
		}

		// Final P-permutation of box selected block
		uint64_t permutedSBlock = permute(boxSelectedBlock, 32, 32, PERMUTATION_TABLE_P);

		// XOR the permuted box selected box with previous left block
		uint64_t newRBlock = previousLBlock ^ permutedSBlock;

		// Reconstruct the block
		previousBlock = (previousRBlock << 32) + newRBlock;
	}

	// Reverse the final encoded block
	uint64_t reversedBlock = (previousBlock << 32) + (previousBlock >> 32);

	// Final permutation
	uint64_t finalBlock = permute(reversedBlock, 64, 64, PERMUTATION_TABLE_IP1);

	return finalBlock;
}

__global__
void encodeK(char* in, char* out, unsigned int size)
{
	unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index >= size)
	{
		return;
	}

	// Parse hex block into 64-bit
	uint64_t block = *((uint64_t*)(in + 16 * index));//std::stoull(message.substr(16 * i, 16), 0, 16);
	// Encode with k1, decode with k2, encode with k3
	uint64_t blockPass1 = processBlock(block, 0, false);
	uint64_t blockPass2 = processBlock(blockPass1, 1, true);
	uint64_t blockPass3 = processBlock(blockPass2, 2, false);

	memcpy(out + 16 * index, &blockPass2, 16);
}

__global__
void decodeK(char* in, char* out, unsigned int size)
{
	unsigned int index = blockIdx.x*blockDim.x + threadIdx.x;
	if (index >= size)
	{
		return;
	}

	// Parse hex block into 64-bit
	uint64_t block = *((uint64_t*)(in + 16 * index));//std::stoull(message.substr(16 * i, 16), 0, 16);
	// Decode with k3, encode with k2, decode with k1
	uint64_t blockPass1 = processBlock(block, 2, true);
	uint64_t blockPass2 = processBlock(blockPass1, 1, false);
	uint64_t blockPass3 = processBlock(blockPass2, 0, true);

	memcpy(out + 16 * index, &blockPass3, 16);
}

char* dev_in=NULL;
char* dev_out=NULL;
cudaError_t err;

std::string TDESCuda::encode(std::string message) {
	this->prepareKeys();

	// Pad the message with zeros
	// Message is hex-encoded -> 16 characters = 64 bits
	int padding = 16 - message.length() % 16;
	if (padding == 16) {
		padding = 0;
	}
	message.append(padding, '0');
	char* msg = new char[message.length()+1];

	// Encode
	// Output strings
	int blockCount = message.length() / 16;
	
	cudaSetDevice(0);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaSetDevice): %s\n", cudaGetErrorString(err));
    }

    cudaFree(0);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaFree(0)): %s\n", cudaGetErrorString(err));
    }

    cudaMalloc((void**)&dev_in, sizeof(char)*message.length());
    cudaMalloc((void**)&dev_out, sizeof(char)*message.length());
	if (err != cudaSuccess)
    {
        printf("cudaError(Malloc): %s\n", cudaGetErrorString(err));
    }
    cudaMemset(dev_in,0,sizeof(char)*message.length());
    cudaMemset(dev_out,0,sizeof(char)*message.length());
	err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaMemset): %s\n", cudaGetErrorString(err));
    }
	
	cudaMemcpy((void*)dev_in, message.c_str(), sizeof(char)*message.length(), cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Memcpy)dev_in: %s\n", cudaGetErrorString(err));
    }

	uint64_t tkeys[3 * 16];
	for (int k = 0; k < 3; k++)
		for (int i = 0; i < 16; i++)
			tkeys[k * 16 + i] = TDESCuda::pKeys[k][i];
	cudaMemcpyToSymbol(ckeys, tkeys, sizeof(uint64_t)*3*16);
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("cudaError(Memcpy)pKeys: %s\n", cudaGetErrorString(err));
	}
	
	unsigned int numThreads = blockCount<=256?blockCount:256;
	unsigned int gridSize = ceil((float)blockCount/256.0);
	encodeK<<< gridSize,numThreads >>>(dev_in,dev_out,blockCount);
	err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(encode): %s\n", cudaGetErrorString(err));
    }
	else
	{
		printf("exit2 encodeK\n");
	}
	cudaMemcpy((void*)msg,dev_out,sizeof(char)*message.length(),cudaMemcpyDeviceToHost);
	err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaMemcpyDeviceToHost): %s\n", cudaGetErrorString(err));
    }
	
	cudaFree(dev_in);
    cudaFree(dev_out);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaFree): %s\n", cudaGetErrorString(err));
    }
	
	std::stringstream hexString;
	msg[message.length()] = 0;
	std::string encodedMessage;
	uint64_t tmp;
	for (int i = 0; i < blockCount; i++)
	{
		tmp = *(((uint64_t*)msg) + i);
		hexString << std::hex << std::setfill('0') << std::setw(16) << tmp;
		encodedMessage.append(hexString.str());
		hexString.clear();
	}
	delete[] msg;
	//std::cout << encodedMessage<<std::endl;
	return encodedMessage;
}

std::string TDESCuda::decode(std::string message) {
	this->prepareKeys();

	// Pad the message with zeros
	// Message is hex-encoded -> 16 characters = 64 bits
	int padding = 16 - message.length() % 16;
	if (padding == 16) {
		padding = 0;
	}
	message.append(padding, '0');

	char* msg = new char[message.length()+1];

	// Encode
	// Output strings
	int blockCount = message.length() / 16;
	
	cudaSetDevice(0);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaSetDevice): %s\n", cudaGetErrorString(err));
    }

    cudaFree(0);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaFree(0)): %s\n", cudaGetErrorString(err));
    }

    cudaMalloc((void**)&dev_in, sizeof(char)*message.length());
    cudaMalloc((void**)&dev_out, sizeof(char)*message.length()*2);
	if (err != cudaSuccess)
    {
        printf("cudaError(Malloc): %s\n", cudaGetErrorString(err));
    }
    cudaMemset(dev_in,0,sizeof(char)*message.length());
    cudaMemset(dev_out,0,sizeof(char)*message.length()*2);
	err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaMemset): %s\n", cudaGetErrorString(err));
    }
	
	cudaMemcpy((void*)dev_in, message.c_str(), sizeof(char)*message.length(), cudaMemcpyHostToDevice);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Memcpy)dev_in: %s\n", cudaGetErrorString(err));
    }

	uint64_t tkeys[3 * 16];
	for (int k = 0; k < 3; k++)
		for (int i = 0; i < 16; i++)
			tkeys[k * 16 + i] = TDESCuda::pKeys[k][i];
	cudaMemcpyToSymbol(ckeys, tkeys, sizeof(uint64_t) * 3 * 16);
	err = cudaGetLastError();
	if (err != cudaSuccess)
	{
		printf("cudaError(Memcpy)pKeys: %s\n", cudaGetErrorString(err));
	}
	
	unsigned int numThreads = blockCount<=256?blockCount:256;
	unsigned int gridSize = ceil((float)blockCount/256.0);
	decodeK<<<gridSize,numThreads>>>(dev_in,dev_out,blockCount);
	
	err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(encode): %s\n", cudaGetErrorString(err));
    }
	
	cudaMemcpy((void*)msg,dev_out,sizeof(char)*message.length(),cudaMemcpyDeviceToHost);
	err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaMemcpyDeviceToHost): %s\n", cudaGetErrorString(err));
    }
	else
	{
		printf("exit2 decodeK\n");
	}
	
	cudaFree(dev_in);
    cudaFree(dev_out);
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaFree): %s\n", cudaGetErrorString(err));
    }
	
	std::stringstream hexString;
	msg[message.length()] = 0;
	std::string decodedMessage;
	uint64_t tmp;
	for (int i = 0; i < blockCount; i++)
	{
		tmp = *(((uint64_t*)msg) + i);
		hexString << std::hex << std::setfill('0') << std::setw(16) << tmp;
		decodedMessage.append(hexString.str());
		hexString.clear();
	}
	delete[] msg;
	//std::cout << decodedMessage << std::endl;
	return decodedMessage;
}
