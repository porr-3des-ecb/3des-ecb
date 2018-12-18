#include "TDESSequential.hpp"
#include "../common/des_defines.hpp"
#include "../common/des_helpers.hpp"

#include <iomanip>
#include <iostream>
#include <string>
#include <sstream>

TDESSequential::~TDESSequential() {}

void TDESSequential::prepareKeys() {
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

uint64_t TDESSequential::processBlock(uint64_t block, int key, bool decode) {
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

std::string TDESSequential::encode(std::string message) {
	this->prepareKeys();

	// Pad the message with zeros
	// Message is hex-encoded -> 16 characters = 64 bits
	int padding = 16 - message.length() % 16;
	if (padding == 16) {
		padding = 0;
	}
	message.append(padding, '0');

	// Encode
	// Output string
	std::string encodedMessage;
	// Iterate over blocks
	int blockCount = message.length() / 16;
	for (int i = 0; i < blockCount; ++i) {
		// Parse hex block into 64-bit
		uint64_t block = std::stoull(message.substr(16 * i, 16), 0, 16);
		// Encode with k1, decode with k2, encode with k3
		uint64_t blockPass1 = this->processBlock(block, 0, false);
		uint64_t blockPass2 = this->processBlock(blockPass1, 1, true);
		uint64_t blockPass3 = this->processBlock(blockPass2, 2, false);

		// Return as hex string
		std::stringstream hexString;
		hexString << std::hex << std::setfill('0') << std::setw(16) << blockPass3;
		encodedMessage.append(hexString.str());
	}

	return encodedMessage;
}

std::string TDESSequential::decode(std::string message) {
	this->prepareKeys();

	// Pad the message with zeros
	// Message is hex-encoded -> 16 characters = 64 bits
	int padding = 16 - message.length() % 16;
	if (padding == 16) {
		padding = 0;
	}
	message.append(padding, '0');

	// Decode
	// Output string
	std::string decodedMessage;
	// Iterate over blocks
	int blockCount = message.length() / 16;
	for (int i = 0; i < blockCount; ++i) {
		// Parse hex block into 64-bit
		uint64_t block = std::stoull(message.substr(16 * i, 16), 0, 16);
		// Decode with k3, encode with k2, decode with k1
		uint64_t blockPass1 = this->processBlock(block, 2, true);
		uint64_t blockPass2 = this->processBlock(blockPass1, 1, false);
		uint64_t blockPass3 = this->processBlock(blockPass2, 0, true);

		// Return as hex string
		std::stringstream hexString;
		hexString << std::hex << std::setfill('0') << std::setw(16) << blockPass3;
		decodedMessage.append(hexString.str());
	}

	return decodedMessage;
}
