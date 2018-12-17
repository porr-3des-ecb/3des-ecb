#include "../TDESInterface.hpp"

class TDESSequential : public TDESInterface {
	private:
		uint64_t pKeys[16];

		void prepareKeys(uint64_t key);

		std::string encodeBlock(std::string block);
		std::string decodeBlock(std::string block);

	public:
		using TDESInterface::TDESInterface;
		~TDESSequential();

		std::string encode(std::string message);
		std::string decode(std::string message);
};
