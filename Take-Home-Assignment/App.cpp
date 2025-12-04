#include "app.h"
#include "Models/Constants.h"

AppState App::g_state;

GLMesh App::gCubeMesh;
GLMesh App::gCuboidMesh;
GLMesh App::gSphereMesh;
GLMesh App::gPyramidMesh;
GLMesh App::gPentagonPrismMesh;
std::vector<GLMesh> App::gDynamicMeshes;   // meshes created from extrusion / revolution
std::vector<SceneObject> App::gObjects;

App::App(Window* win2d, Window* win3d) : _win2d(win2d), _win3d(win3d) {}

App::~App() {
	glfwDestroyWindow(this->_win3d->getWindowInstance());
	glfwDestroyWindow(this->_win2d->getWindowInstance());
	glfwTerminate();

	delete _win2d;
	delete _win3d;
}

void App::initialize() {
	if (!glfwInit()) {
		throw "Failed to init GLFW!";
	}

	App::gDynamicMeshes.reserve(1000); // initialize capaciyt to avoid reallocation data loss
	App::gObjects.reserve(1000);

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

	// 1. Create the FIRST window (2D)
	this->_win2d->initialize();

	// 3. Create the 3D window and share context with 2D
	this->_win3d->initialize(this->_win2d->getWindowInstance());

	glfwMakeContextCurrent(this->_win2d->getWindowInstance());
}


void App::run() {
	auto t0 = std::chrono::steady_clock::now();
	int frames = 0;

	while (!glfwWindowShouldClose(this->_win2d->getWindowInstance()) &&
		!glfwWindowShouldClose(this->_win3d->getWindowInstance())) {

		this->_win2d->run();
		this->_win3d->run();

		frames++;

		auto t1 = std::chrono::steady_clock::now();
		if (std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count() >= 1) {

			std::cout << "FPS: " << frames << ", # objects: " << App::gObjects.size() << std::endl;

			frames = 0; t0 = t1;
		}

		glfwPollEvents();
	}
}