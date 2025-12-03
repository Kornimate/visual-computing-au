#pragma once
#include <opencv2/opencv.hpp>
#include <glad/glad.h>

struct AppState {
	bool drawing = false;
	double lastX = 0.0;
	double lastY = 0.0;
	bool needUpdate = true;
	cv::Mat canvas;

	GLuint tex = 0;
	GLuint quadVAO = 0;
	GLuint quadVBO = 0;
	GLuint quadEBO = 0;
};