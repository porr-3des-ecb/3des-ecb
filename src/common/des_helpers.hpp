#ifndef _DES_HELPERS_HPP
#define _DES_HELPERS_HPP

#include <cstdint>

namespace TDES {
	uint64_t permute(const uint64_t in, const int inSize, const int outSize, const uint8_t * table);
	uint64_t lshift(const uint64_t in, const int bitSize);

	void printBit(const uint64_t in, const int bitSize = 64);
}

#endif
