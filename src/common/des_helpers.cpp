#include "des_helpers.hpp"

#include <bitset>
#include <iostream>

namespace TDES {

uint64_t permute(const uint64_t in, const int inSize, const int outSize, const uint8_t * table) {
	uint64_t out = 0;

	for (int i = 0; i < outSize; ++i) {
		out = out << 1;
		out += (in >> (inSize - table[i])) & 0x01;
	}

	return out;
}

uint64_t lshift(const uint64_t in, const int bitSize) {
	uint8_t t = !!(in & (0x1 << (bitSize - 1)));
	uint64_t mask = 0xffffffff >> (64 - bitSize);
	uint64_t shifted = (in << 1) + t;
	return shifted & mask;
}

void printBit(const uint64_t in, const int bitSize) {
	int maxByte = bitSize / 8 - 1;
	for (int i = maxByte; i >= 0; --i) {
		std::bitset<8> x(in >> (8 * i));
		std::cout << x << " ";
	}
	std::cout << std::endl;
}

// namespace TDES
}
