#ifndef _TDESINTERFACE_HPP
#define _TDESINTERFACE_HPP

#include <cstdint>
#include <string>

class TDESInterface {
	private:
		uint64_t keys[3];

		bool verifyKey(int keyIndex);

	public:
		TDESInterface(std::string key);
		TDESInterface(uint64_t keys[3]);
		virtual ~TDESInterface();

		virtual std::string encode(std::string message) = 0;
		virtual std::string decode(std::string message) = 0;
};

#endif
