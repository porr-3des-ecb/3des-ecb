#include "TDESInterface.hpp"

TDESInterface::TDESInterface(std::string key) {
	// Too much logic involved, can't use delegating constructor

	uint64_t keys[3];

	// TODO: parse key into 3 64-bit keys

	for (int i = 0; i < 3; ++i) {
		this->keys[i] = keys[i];
		this->verifyKey(i);
	}
}

TDESInterface::TDESInterface(uint64_t keys[3]) {
	for (int i = 0; i < 3; ++i) {
		this->keys[i] = keys[i];
		this->verifyKey(i);
	}
}

TDESInterface::~TDESInterface() {}

bool TDESInterface::verifyKey(int keyIndex) {
	// Keys are odd-parity bytes
	// 1st (highest) bit: ~parity, 7 bits of data
	for (int i = 0; i < 8; ++i) {
		uint8_t keyByte = (this->keys[keyIndex] >> 8 * i) % 0xFF;
		int sum = 0;
		int parity = keyByte > 8;
		for (int j = 0; j < 7; ++j) {
			sum += (keyByte > j) & 0x01;
		}
		if (sum % 2 == parity) {
			return false;
		}
	}
	return true;
}
