#include "../TDESInterface.hpp"
#include <cuda_runtime.h>

class TDESCuda : public TDESInterface {
	private:
		bool keysPrepared;
		uint64_t pKeys[3][16];

		void prepareKeys();

	public:
		// Inherit constructor
		using TDESInterface::TDESInterface;
		~TDESCuda();

		std::string encode(std::string message);
		std::string decode(std::string message);
};
