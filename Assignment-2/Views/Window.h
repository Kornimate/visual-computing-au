#pragma once
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>
#include "../Services/LoggerService.h"
#include "../Models/Filters.h"


class Window {
private:
	GLFWwindow* _win;
	int _initialW;
	int _initialH;
	LoggerService* _logger;
	cv::VideoCapture _cap;
	cv::Mat _frameBGR;
	int _camW;
	int _camH;
	GLuint _tex;
	GLuint _vao;
	GLuint _vbo;
	GLuint _ebo;
	GLint _uTransformLoc;
	GLint _uTexLoc;
	GLint _uScaleLoc;
	GLint _uFilterTypeLoc;
	GLint _uPixelBlockLoc;
	GLint _uRedThreshLoc;
	GLuint _prog;
	bool _useGPU; // CPU/GPU toggle
	int _filter;
	int _pixelBlock; // default block size for Pixelation Filter
	float _redThresh;
	cv::Mat _cpuOutBGR;
	AppState _state;

	static void cursorPosCB(GLFWwindow* w, double x, double y);
	static void mouseButtonCB(GLFWwindow* w, int button, int action, int mods);
	static void Window::scrollCB(GLFWwindow* w, double /*xoff*/, double yoff);

public:
	Window(int initialW, int inititalH);
	~Window();
	int init();
	void run();
};