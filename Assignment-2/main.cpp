// Usage:
//  Filters:
//   [Key 1] None 
//   [Key 2] Pixelate   
//   [Key 3] SinCity   
//   [Key 4] Comic  
//  Runtime:
//   [Key G] GPU path   [Key C] CPU path
//  Custom controls for filters:
//   [Key Down Arrow] decrease pixel block size, [Key Up Arrow] increase pixel block size
//  Transforms:
//   [Mouse Left-drag]: translate (pan)
//   [Mouse Right-drag] OR [Key Shift+ Mouse Left-drag] rotate
//   [Mouse wheel] zoom
//   [KeyR] reset transform
//   [Key Esc] Quit

#include "Services/MatrixService.h"
#include "Services/TransformService.h"
#include "Views/Window.h"
#include "Views/Shader.h"
#include "Models/AppConstans.hpp"
#include "Models/Filters.h"

int main() {
	// Init GLFW
	if (!glfwInit()) {
		std::cerr << "Failed to init GLFW" << std::endl;
		return EXIT_FAILURE;
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

#ifdef __APPLE__ // extra setting for macos users
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

	Window* window = new Window(AppConstants::WIN_W, AppConstants::WIN_H);


	// test resolutions : 320, 180; 960, 540; 1280, 720;
	if (!window->init()) {
		std::cerr << "Failed to initialize the window!" << std::endl;
		delete window;
		return EXIT_FAILURE;
	}

	window->run();

	delete window;

	return EXIT_SUCCESS;
}
