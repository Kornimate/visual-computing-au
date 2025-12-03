#pragma once
#include <GLFW/glfw3.h>

class Window {
public:
	virtual void run() = 0;
	virtual void initialize() = 0;
	virtual GLFWwindow* getWindowInstance() = 0;
};