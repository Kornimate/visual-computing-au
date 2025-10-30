#pragma once
#include <opencv2/opencv.hpp>
#include "MatrixService.h"


// Interactive App State
struct AppState {
	// Transform in UV space (u,v in [0,1])
	float rot = 0.0f;     // radians
	float scale = 1.0f;   // uniform
	float tx = 0.0f;      // translation (UV units)
	float ty = 0.0f;      // translation (UV units)

	bool draggingPan = false, draggingRot = false;
	double lastX = 0.0, lastY = 0.0;

	// framebuffer size & letterbox scale for mouse mapping
	int winW = 1280, winH = 720;
	float sx = 1.f, sy = 1.f;

	// camera size
	int camW = 1, camH = 1;
};

class TransformService {
public:
	static Mat3 buildUvForward(const AppState& st);
	static void uvForward_to_cvInverse2x3(const AppState& st, int camW, int camH, cv::Mat& cv2x3Inv);
};