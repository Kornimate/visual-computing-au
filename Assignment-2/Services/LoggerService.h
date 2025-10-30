#pragma once
#include "../Services/TransformService.h"

class LoggerService {
public:
	void LogControls();
	void LogStatusOfApp(int frames, int pixelBlock, int filter, bool useGPU, AppState& state);
	void LogCameraResolution(int camW, int camH);
};