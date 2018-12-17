#include "TDESSequential.hpp"
#include "../des_defines.hpp"

TDESSequential::~TDESSequential() {}

void TDESSequential::prepareKeys(uint64_t key) {
	// Initial key permutation
	uint64_t permutedKey = 0;
	for (int i = 0; i < 56; ++i) {
		permutedKey = permutedKey << 1;
		permutedKey += (key >> (TDES::PC1[i] - 1)) & 0x01;
	}

	// Left/right half-key rotations
	uint64_t previousKey = permutedKey;
	for (int i = 0; i < 16; ++i) {
		// Split the key in 2
		uint64_t lKey = (previousKey >> 28) & 0x0fffffff;
		uint64_t rKey = previousKey & 0x0fffffff;

		// Left shift and rotate the key parts the correct number of times
		int shifts = (i == 0 || i == 1 || i == 8 || i == 15) ? 1 : 2;
		for (int j = 0; j < shifts; ++j) {
			uint64_t t = (lKey & 0x80000000) ? 1 : 0;
			lKey = ((lKey << 1) + t) & 0x0fffffff;
			t = (rKey & 0x80000000) ? 1 : 0;
			rKey = ((rKey << 1) + t) & 0x0fffffff;
		}

		// Save the key
		previousKey = (lKey << 28) + rKey;
		this->pKeys[i] = previousKey;
	}

	// Final key permutation
	for (int i = 0; i < 16; ++i) {
		uint64_t finalKey = 0;
		for (int j = 0; j < 48; ++j) {
			finalKey = finalKey << 1;
			finalKey += (this->pKeys[i] >> (TDES::PC2[i] - 1)) & 0x01;
		}
		this->pKeys[i] = finalKey;
	}
}

std::string TDESSequential::encodeBlock(std::string block) {}
std::string TDESSequential::decodeBlock(std::string block) {}
std::string TDESSequential::encode(std::string message) {}
std::string TDESSequential::decode(std::string message) {}
