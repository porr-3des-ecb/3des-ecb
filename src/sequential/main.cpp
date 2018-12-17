#include <iostream>
#include <fstream>
#include <string>

#include "TDESSequential.hpp"

int main(int argc, char const *argv[]) {
	if (argc != 4) {
		std::cout << "Usage: " << argv[0] << " <inputFile> <outputFile> <key>" << std::endl;
		return 1;
	}

	std::ifstream iFile(argv[1]);
	if (!iFile.good()) {
		std::cout << "Opening file " << argv[1] << " for reading failed" << std::endl;
		return 2;
	}

	std::ofstream oFile(argv[2]);
	if (!oFile.good()) {
		std::cout << "Opening file " << argv[2] << " for writing failed" << std::endl;
		return 2;
	}

	std::string key = argv[3];

	/// TODO: start time measurements

	TDESSequential sequentialEncoder(key);

	/// TODO: end and display time measurements

	iFile.close();
	oFile.close();

	return 0;
}
