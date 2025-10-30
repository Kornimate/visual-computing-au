#include <iostream>
#include <GLFW/glfw3.h>
#include "LoggerService.h"
#include "../Services/TransformService.h"

void LoggerService::LogControls() {
	std::cout << "GL: " << glGetString(GL_VERSION) << "\n";
	std::cout << "Controls: [1] Pixelate  [2] SinCity [3] Comic [0] None  |  [G] GPU  [C] CPU"
		<< "  |  Left-drag: pan  Right-drag/Shift+L: rotate  Wheel: zoom  [R]: reset  [Esc]: Quit\n";
}

void LoggerService::LogStatusOfApp(int frames, int pixelBlock, int filter, bool useGPU, AppState& state) {
	std::cout << "FPS: " << frames << (useGPU ? " [GPU]" : " [CPU]")
		<< " | Filter: " << (filter == 0 ? "None" : (filter == 1 ? "Pixelate" : (filter == 2 ? "SinCity" : "Comic")))
		<< " | Block: " << pixelBlock
		<< " | rot(deg): " << (state.rot * 180.0 / 3.14159265)
		<< " | scale: " << state.scale
		<< " | tx,ty: " << state.tx << "," << state.ty
		<< std::endl;
}

void LoggerService::LogCameraResolution(int camW, int camH) {
	std::cout << "Camera resolution: " << camW << " x " << camH << std::endl;
}