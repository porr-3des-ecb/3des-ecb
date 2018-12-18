#ifndef _DES_HELPERS_HPP
#define _DES_HELPERS_HPP

#include <cstdint>
#include <string>
#include <algorithm>
#include <cctype>
#include <locale>

namespace TDES {
	uint64_t permute(const uint64_t in, const int inSize, const int outSize, const uint8_t * table);
	uint64_t lshift(const uint64_t in, const int bitSize);

	void printBit(const uint64_t in, const int bitSize = 64);

	// String trimming helpers, from https://stackoverflow.com/a/217605/3217805
	static inline void ltrim(std::string &s) {
	    s.erase(s.begin(), std::find_if(s.begin(), s.end(), [](int ch) {
	        return !std::isspace(ch);
	    }));
	}
	static inline void rtrim(std::string &s) {
	    s.erase(std::find_if(s.rbegin(), s.rend(), [](int ch) {
	        return !std::isspace(ch);
	    }).base(), s.end());
	}
	static inline void trim(std::string &s) {
	    ltrim(s);
	    rtrim(s);
	}
}

#endif
