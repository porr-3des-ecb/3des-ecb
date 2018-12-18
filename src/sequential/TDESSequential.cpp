#include "TDESSequential.hpp"
#include "../common/des_defines.hpp"
#include "../common/des_helpers.hpp"

#include <string>
#include <sstream>
#include <iostream>
#include <bitset>

TDESSequential::~TDESSequential() {}

void TDESSequential::prepareKeys(uint64_t key) {
	// Initial key permutation
	uint64_t permutedKey = TDES::permute(key, 64, 56, TDES::PERMUTATION_TABLE_PC1);

	// Left/right half-key rotations
	uint64_t previousKey = permutedKey;
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
		this->pKeys[i] = previousKey;
	}

	// Final key permutation
	for (int i = 0; i < 16; ++i) {
		this->pKeys[i] = TDES::permute(this->pKeys[i], 56, 48, TDES::PERMUTATION_TABLE_PC2);
	}
}

uint64_t TDESSequential::encodeBlock(uint64_t block) {
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
		uint64_t xoredBlock = this->pKeys[i] ^ extendedBlock;

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

uint64_t TDESSequential::decodeBlock(uint64_t block) {}

std::string TDESSequential::encode(std::string message) {
	// TODO: block-divide
	// TODO: triplify the DES

	this->prepareKeys(this->keys[0]);

	uint64_t block = std::stoull(message.substr(0, 16), 0, 16);
	uint64_t encodedBlock = this->encodeBlock(block);
	std::stringstream hexSS;
	hexSS << std::hex << encodedBlock;
	return hexSS.str();
}

std::string TDESSequential::decode(std::string message) {}
