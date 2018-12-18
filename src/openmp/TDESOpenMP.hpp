#include "../TDESInterface.hpp"

class TDESOpenMP : public TDESInterface {
	private:
		bool keysPrepared;
		uint64_t pKeys[3][16];

		void prepareKeys();
		uint64_t processBlock(uint64_t block, int key = 0, bool decode = false);

	public:
		// Inherit constructor
		using TDESInterface::TDESInterface;
		~TDESOpenMP();

		std::string encode(std::string message);
		std::string decode(std::string message);
};
