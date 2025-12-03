#include "FileService.h"
#include <fstream>
#include <iostream>

std::string FileService::ReadFileContent(std::string fileName) {
	std::string fileContent = "";
	std::ifstream fin = std::ifstream(fileName, std::ios::in);

	if (!fin.is_open()) {
		std::cerr << "Failed to open file: " << fileName << std::endl;
		return "";
	}

	std::string currentLine = "";
	while (std::getline(fin, currentLine)) {
		fileContent += "\n" + currentLine;
	}

	fin.close();

	return fileContent;
}