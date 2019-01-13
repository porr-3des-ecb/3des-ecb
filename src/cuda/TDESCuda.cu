#include "TDESCuda.hpp"
#include "../common/des_defines.hpp"
#include "../common/des_helpers.hpp"

#include <iomanip>
#include <iostream>
#include <string>
#include <sstream>
#include <cuda_runtime.h>
//#include <helper_functions.h>
//#include <helper_cuda.h>

TDESCuda::~TDESCuda() {}

void TDESCuda::prepareKeys() {
	for (int k = 0; k < 3; ++k) {
		uint64_t key = this->keys[k];

		// Initial key permutation
		uint64_t permutedKey = TDES::permute(key, 64, 56, TDES::PERMUTATION_TABLE_PC1);

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
			this->pKeys[k][i] = TDES::permute(pKeys[i], 56, 48, TDES::PERMUTATION_TABLE_PC2);
		}
	}
}

__device__
uint64_t TDESCuda::processBlock(uint64_t block, int key, bool decode) {
	// Initial permutation
	uint64_t permutedBlock = TDES::permute(block, 64, 64, TDES::PERMUTATION_TABLE_IP);

	// Encoding (16 passes)
	uint64_t previousBlock = permutedBlock;
	for (int i = 0; i < 16; ++i) {
		// Split previous block in 2
		uint64_t previousLBlock = (previousBlock >> 32) & 0xffffffff;
		uint64_t previousRBlock = previousBlock & 0xffffffff;

		// Extend it (32b -> 48b)
		uint64_t extendedBlock = TDES::permute(previousRBlock, 32, 48, TDES::SELECTION_TABLE_E);

		// XOR the extended block with a prepared key
		uint64_t pKey = this->pKeys[key][i];
		if (decode) {
			pKey = this->pKeys[key][15 - i];
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
			boxSelectedBlock += TDES::SELECTION_BOX[j][16 * row + column];
		}

		// Final P-permutation of box selected block
		uint64_t permutedSBlock = TDES::permute(boxSelectedBlock, 32, 32, TDES::PERMUTATION_TABLE_P);

		// XOR the permuted box selected box with previous left block
		uint64_t newRBlock = previousLBlock ^ permutedSBlock;

		// Reconstruct the block
		previousBlock = (previousRBlock << 32) + newRBlock;
	}

	// Reverse the final encoded block
	uint64_t reversedBlock = (previousBlock << 32) + (previousBlock >> 32);

	// Final permutation
	uint64_t finalBlock = TDES::permute(reversedBlock, 64, 64, TDES::PERMUTATION_TABLE_IP1);

	return finalBlock;
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
	char* msg = new char[message.length()];

	// Encode
	// Output strings
	int blockCount = message.length() / 16;
	
	checkCudaErrors(cudaSetDevice(0));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaSetDevice): %s\n", cudaGetErrorString(err));
    }

    checkCudaErrors(cudaFree(0));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaFree(0)): %s\n", cudaGetErrorString(err));
    }

    checkCudaErrors(cudaMalloc((void**)&dev_in, sizeof(char)*message.length()));
    checkCudaErrors(cudaMalloc((void**)&dev_out, sizeof(char)*message.length()*2));
	if (err != cudaSuccess)
    {
        printf("cudaError(Malloc): %s\n", cudaGetErrorString(err));
    }
    checkCudaErrors(cudaMemset(dev_in,0,sizeof(char)*message.length()));
    checkCudaErrors(cudaMemset(dev_out,0,sizeof(char)*message.length()*2));
	err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaMemset): %s\n", cudaGetErrorString(err));
    }
	
	checkCudaErrors(cudaMemcpy((void*)dev_in, message.c_str(), sizeof(char)*message.length(), cudaMemcpyHostToDevice));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Memcpy): %s\n", cudaGetErrorString(err));
        return;
    }
	
	uint numThreads = blockCount<=256?blockCount:256;
	uint gridSize = ceil(blockCount,256);
	encode<<<gridSize,numThreads>>>(dev_in,dev_out,blockCount);
	err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(encode): %s\n", cudaGetErrorString(err));
    }
	checkCudaErrors(cudaMemcpy((void*)msg,dev_out,sizeof(char)*message.length()*2,cudaMemcpyDeviceToHost));
	err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaMemcpyDeviceToHost): %s\n", cudaGetErrorString(err));
    }
	
	checkCudaErrors(cudaFree(dev_headerList));
    checkCudaErrors(cudaFree(dev_headerCounter));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaFree): %s\n", cudaGetErrorString(err));
    }
	

	std::string encodedMessage(msg);
	delete[] msg;
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

	char* msg = new char[message.length()];

	// Encode
	// Output strings
	int blockCount = message.length() / 16;
	
	checkCudaErrors(cudaSetDevice(0));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaSetDevice): %s\n", cudaGetErrorString(err));
    }

    checkCudaErrors(cudaFree(0));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaFree(0)): %s\n", cudaGetErrorString(err));
    }

    checkCudaErrors(cudaMalloc((void**)&dev_in, sizeof(char)*message.length()));
    checkCudaErrors(cudaMalloc((void**)&dev_out, sizeof(char)*message.length()*2));
	if (err != cudaSuccess)
    {
        printf("cudaError(Malloc): %s\n", cudaGetErrorString(err));
    }
    checkCudaErrors(cudaMemset(dev_in,0,sizeof(char)*message.length()));
    checkCudaErrors(cudaMemset(dev_out,0,sizeof(char)*message.length()*2));
	err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaMemset): %s\n", cudaGetErrorString(err));
    }
	
	checkCudaErrors(cudaMemcpy((void*)dev_in, message.c_str(), sizeof(char)*message.length(), cudaMemcpyHostToDevice));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(Memcpy): %s\n", cudaGetErrorString(err));
        return;
    }
	
	uint numThreads = blockCount<=256?blockCount:256;
	uint gridSize = ceil(blockCount,256);
	encode<<<gridSize,numThreads>>>(dev_in,dev_out,blockCount);
	err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(encode): %s\n", cudaGetErrorString(err));
    }
	checkCudaErrors(cudaMemcpy((void*)msg,dev_out,sizeof(char)*message.length()*2,cudaMemcpyDeviceToHost));
	err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaMemcpyDeviceToHost): %s\n", cudaGetErrorString(err));
    }
	
	checkCudaErrors(cudaFree(dev_in));
    checkCudaErrors(cudaFree(dev_out));
    err = cudaGetLastError();
    if (err != cudaSuccess)
    {
        printf("cudaError(cudaFree): %s\n", cudaGetErrorString(err));
    }
	

	std::string decodedMessage(msg);
	return decodedMessage;
}
