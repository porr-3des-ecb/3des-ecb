#include "../TDESInterface.hpp"

class TDESSequential : public TDESInterface {
	private:

	public:
		using TDESInterface::TDESInterface;
		~TDESSequential();

		std::string encode(std::string message);
		std::string decode(std::string message);
};
