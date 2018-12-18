#include "../TDESInterface.hpp"

class TDESSequential : public TDESInterface {
	private:
		uint64_t pKeys[16];

		void prepareKeys(uint64_t key);

		uint64_t encodeBlock(uint64_t block);
		uint64_t decodeBlock(uint64_t block);

	public:
		using TDESInterface::TDESInterface;
		~TDESSequential();

		std::string encode(std::string message);
		std::string decode(std::string message);
};
