#include "app.h"

App::App(Window* win2d, Window* win3d) : _win2d(win2d), _win3d(win3d) {}

App::~App() {
	delete _win2d;
	delete _win3d;
}

void App::initialize() {
	if (!glfwInit()) {
		std::cerr << "Failed to init GLFW.\n";
		throw "Error while loading GLFW!";
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	this->_win2d->initialize();
	this->_win3d->initialize();
}

void App::run() {
	this->_win2d->run();
	this->_win3d->run();
}