#include <iostream>
#include <fstream>
#include <string>

#include "TDEScuda.cuh"
#include "../common/des_helpers.hpp"
#include <chrono>
#include <sstream>
#include <cuda_runtime.h>

int main(int argc, char const *argv[]) {
	if (argc != 4) {
		std::cout << "Usage: " << argv[0] << " <inputFile> <outputFile> <key>" << std::endl;
		return 1;
	}

	std::ifstream inputFile(argv[1], std::ifstream::in | std::ifstream::binary);
	if (!inputFile.good()) {
		std::cout << "Opening file " << argv[1] << " for reading failed" << std::endl;
		return 2;
	}

	std::ofstream outputFile(argv[2], std::ofstream::out | std::ifstream::binary);
	if (!outputFile.good()) {
		std::cout << "Opening file " << argv[2] << " for writing failed" << std::endl;
		return 2;
	}

	std::string key = argv[3];

	// Prepare input data
	std::stringstream inDataStream;
	inDataStream << inputFile.rdbuf();
	inputFile.close();
	std::string originalMessage = inDataStream.str();
	TDES::trim(originalMessage);

	// Prepare encoder
	TDESCuda sequentialEncoder(key);

	// Start time measurement
	auto start = std::chrono::high_resolution_clock::now();

	// Encode and decode
	std::string encodedMessage = sequentialEncoder.encode(originalMessage);
	std::string decodedMessage = sequentialEncoder.decode(encodedMessage);

	// End time measurement and calculate the difference
	auto end = std::chrono::high_resolution_clock::now();
	auto timeSpan = std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(end - start);
	double calculationTime = timeSpan.count();

	// Verify original data with encoded/decoded one
	if (originalMessage.compare(decodedMessage) != 0) {
		std::cout << "Original and encoded/decoded messages differ!" << std::endl;
		std::cout << "Original:  " << originalMessage << std::endl;
		std::cout << "Processed: " << decodedMessage << std::endl;
	}

	// Display processing time
	std::cout << "Processing time (ms): " << std::dec << calculationTime << std::endl;

	// Save data to output file
	// TODO
	outputFile.close();

	return 0;
}
