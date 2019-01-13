#pragma once
#include <cuda_runtime.h>

__global__
void encodeK(char* in, char* out,unsigned int size);

__global__
void decodeK(char* in, char* out,unsigned int size);