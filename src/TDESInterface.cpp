#include "TDESInterface.hpp"

#include <iostream>
#include <stdexcept>
#include <string>

TDESInterface::TDESInterface(std::string key) {
	// Parse hex strings
	this->keys[0] = std::stoull(key.substr(0, 16), 0, 16);
	this->keys[1] = std::stoull(key.substr(16, 16), 0, 16);
	this->keys[2] = std::stoull(key.substr(32, 16), 0, 16);

	// Save and verify
	for (int i = 0; i < 3; ++i) {
		if(!this->verifyKey(i)) {
			// throw std::invalid_argument("Bad key");
			std::cout << "Warning: invalid key!" << std::endl;
		}
	}
}

TDESInterface::TDESInterface(uint64_t keys[3]) {
	for (int i = 0; i < 3; ++i) {
		this->keys[i] = keys[i];
		if(!this->verifyKey(i)) {
			// throw std::invalid_argument("Bad key");
			std::cout << "Warning: invalid key!" << std::endl;
		}
	}
}

TDESInterface::~TDESInterface() {}

bool TDESInterface::verifyKey(int keyIndex) {
	// Keys are odd-parity bytes
	// 1st (highest) bit: ~parity, 7 bits of data
	for (int i = 0; i < 8; ++i) {
		uint8_t keyByte = (this->keys[keyIndex] >> (8 * i)) & 0xFF;

		int sum = 0;
		int parity = keyByte >> 8;
		for (int j = 0; j < 7; ++j) {
			sum += (keyByte >> j) & 0x01;
		}

		if (sum % 2 == parity) {
			return false;
		}
	}
	return true;
}
