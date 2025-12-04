#pragma once
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "../Models/SceneObject.h"
#include "../Models/Constants.h"
#include "window.h"

class Window;

class Window3D : public Window {
public:
	Window3D();
	void run();
	void initialize(GLFWwindow* win2D = nullptr);
	GLFWwindow* getWindowInstance();
	
	static void Window3D::getRayFromMouse(double mouseX, double mouseY, int screenW, int screenH, const glm::mat4& projection, const glm::mat4& view, glm::vec3& outOrigin, glm::vec3& outDirection);
	static bool rayIntersectsAABB(const glm::vec3& origin, const glm::vec3& dir, const glm::vec3& minBound, const glm::vec3& maxBound, float& tHit);
	static glm::mat4 getObjectModelMatrix(const SceneObject& obj);
	static bool rayIntersectsObject(const SceneObject& obj, const glm::vec3& rayOrigin, const glm::vec3& rayDir, float& tHit);
	
	void processInput3D(GLFWwindow* window);
	static void framebuffer_size_callback_3D(GLFWwindow* window, int width, int height);
	static void key_callback_3D(GLFWwindow* window, int key, int scancode, int action, int mods);
	static void mouse_button_callback_3D(GLFWwindow* window, int button, int action, int mods);
	static void cursor_position_callback_3D(GLFWwindow* window, double xpos, double ypos);
	static void scroll_callback_3D(GLFWwindow* window, double xoffset, double yoffset);

private:
	GLFWwindow* _win3D;
	GLuint _gProg3D;

	static float _gDeltaTime;
	static float _gLastFrame;

	static glm::vec3 _gCameraPos;
	static glm::vec3 _gCameraFront;
	static glm::vec3 _gCameraUp;

	static float _gYaw;
	static float _gPitch;

	static bool _gTransformMode;
	static bool _gLeftMouseDown;
	static bool _gRightMouseDown;
	static bool _gIsDraggingObject;
	static bool _gFirstMouse3D;
	static float _gLastX3D;
	static float _gLastY3D;
	static int _g3DWidth;
	static int _g3DHeight;

	static int _gSelectedObject;
	static glm::vec3 _gDragPlanePoint;
	static glm::vec3 _gDragPlaneNormal;
	static bool _gHasDragPlane;
};