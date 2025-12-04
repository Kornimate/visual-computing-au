#pragma once
#include "window.h"
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "../Models/SceneObject.h"
#include "../Models/Constants.h"

class Window;

class Window3D : public Window {
public:
	void run();
	void initialize(GLFWwindow* win2D = nullptr);
	GLFWwindow* getWindowInstance();
	
	static void Window3D::getRayFromMouse(double mouseX, double mouseY, int screenW, int screenH, const glm::mat4& projection, const glm::mat4& view, glm::vec3& outOrigin, glm::vec3& outDirection);
	static bool rayIntersectsAABB(const glm::vec3& origin, const glm::vec3& dir, const glm::vec3& minBound, const glm::vec3& maxBound, float& tHit);
	static glm::mat4 getObjectModelMatrix(const SceneObject& obj);
	static bool rayIntersectsObject(const SceneObject& obj, const glm::vec3& rayOrigin, const glm::vec3& rayDir, float& tHit);
	
	void processInput3D(GLFWwindow* window);
	void framebuffer_size_callback_3D(GLFWwindow* window, int width, int height);
	void key_callback_3D(GLFWwindow* window, int key, int scancode, int action, int mods);
	void mouse_button_callback_3D(GLFWwindow* window, int button, int action, int mods);
	void cursor_position_callback_3D(GLFWwindow* window, double xpos, double ypos);
	void scroll_callback_3D(GLFWwindow* window, double xoffset, double yoffset);

private:
	GLFWwindow* _win3D;
	GLuint _gProg3D = 0;

	float _gDeltaTime = 0.0f;
	float _gLastFrame = 0.0f;

	glm::vec3 _gCameraPos = glm::vec3(0.0f, 0.0f, 6.0f);
	glm::vec3 _gCameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
	glm::vec3 _gCameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

	float _gYaw = -90.0f;
	float _gPitch = 0.0f;

	bool  _gTransformMode = false;
	bool  _gLeftMouseDown = false;
	bool  _gRightMouseDown = false;
	bool  _gIsDraggingObject = false;
	bool  _gFirstMouse3D = true;
	float _gLastX3D = 400.0f;
	float _gLastY3D = 300.0f;
	int   _g3DWidth = AppConstants::DRAW_W;
	int   _g3DHeight = AppConstants::DRAW_H;

	int   _gSelectedObject = -1;
	glm::vec3 _gDragPlanePoint = glm::vec3(0.0f);
	glm::vec3 _gDragPlaneNormal = glm::vec3(0.0f, 0.0f, -1.0f);
	bool  _gHasDragPlane = false;
};