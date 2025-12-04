#pragma once
#include "window.h"
#include "../Models/AppState.h"
#include "../Models/SceneObject.h"
#include <glm/glm.hpp>

class Window2D : public Window {
public:
	Window2D();
	void run();
	void initialize(GLFWwindow* win2D = nullptr);
	GLFWwindow* getWindowInstance();
private:
	static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
	static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos);
	static void key_callback_2D(GLFWwindow* window, int key, int scancode, int action, int mods);
	static glm::vec3 randomBrightColor();

	GLuint _gProg2D;
	GLFWwindow* _win2D;
};