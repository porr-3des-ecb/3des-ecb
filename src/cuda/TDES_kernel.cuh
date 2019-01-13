#pragma once

__global__
void encode(char* in, char* out,unsigned int size);

__global__
void decode(char* in, char* out,unsigned int size);