#include "window3D.h"
#include "../App.h"
#include "../Services/FileService.h"
#include "../Services/ShaderService.h"
#include "../Services/PredefinedShapeBuilderService.h"
#include "../Services/GPUConvertService.h"
#include "../Models/Mesh.h"
#include <iostream>

float Window3D::_gDeltaTime = 0.0f;
float Window3D::_gLastFrame = 0.0f;

glm::vec3 Window3D::_gCameraPos = glm::vec3(0.0f, 0.0f, 6.0f);
glm::vec3 Window3D::_gCameraFront = glm::vec3(0.0f, 0.0f, -1.0f);
glm::vec3 Window3D::_gCameraUp = glm::vec3(0.0f, 1.0f, 0.0f);

float Window3D::_gYaw = -90.0f;
float Window3D::_gPitch = 0.0f;

bool Window3D::_gTransformMode = false;
bool Window3D::_gLeftMouseDown = false;
bool Window3D::_gRightMouseDown = false;
bool Window3D::_gIsDraggingObject = false;
bool Window3D::_gFirstMouse3D = true;
float Window3D::_gLastX3D = 400.0f;
float Window3D::_gLastY3D = 300.0f;
int Window3D::_g3DWidth = AppConstants::DRAW_W;
int Window3D::_g3DHeight = AppConstants::DRAW_H;

int Window3D::_gSelectedObject = -1;
glm::vec3 Window3D::_gDragPlanePoint = glm::vec3(0.0f);
glm::vec3 Window3D::_gDragPlaneNormal = glm::vec3(0.0f, 0.0f, -1.0f);
bool Window3D::_gHasDragPlane = false;

Window3D::Window3D() {
	this->_win3D = nullptr;
	this->_gProg3D = 0;
}

void Window3D::run() {
	float currentFrame = (float)glfwGetTime();
	this->_gDeltaTime = currentFrame - this->_gLastFrame;
	this->_gLastFrame = currentFrame;

	glfwMakeContextCurrent(this->_win3D);
	int fbw3, fbh3;
	glfwGetFramebufferSize(this->_win3D, &fbw3, &fbh3);
	glViewport(0, 0, fbw3, fbh3);

	processInput3D(this->_win3D);

	glClearColor(0.1f, 0.12f, 0.15f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glUseProgram(this->_gProg3D);

	glm::mat4 view = glm::lookAt(this->_gCameraPos, this->_gCameraPos + this->_gCameraFront, this->_gCameraUp);
	glm::mat4 projection = glm::perspective(glm::radians(45.0f),
		(float)fbw3 / (float)fbh3,
		0.1f, 100.0f);

	GLint locView = glGetUniformLocation(this->_gProg3D, "view");
	GLint locProj = glGetUniformLocation(this->_gProg3D, "projection");
	GLint locModel = glGetUniformLocation(this->_gProg3D, "model");
	GLint locColor = glGetUniformLocation(this->_gProg3D, "objectColor");

	glUniformMatrix4fv(locView, 1, GL_FALSE, glm::value_ptr(view));
	glUniformMatrix4fv(locProj, 1, GL_FALSE, glm::value_ptr(projection));

	for (size_t i = 0; i < App::gObjects.size(); ++i) {
		SceneObject& obj = App::gObjects[i];
		glm::mat4 model = getObjectModelMatrix(obj);
		glUniformMatrix4fv(locModel, 1, GL_FALSE, glm::value_ptr(model));
		glUniform3fv(locColor, 1, glm::value_ptr(obj.color));

		glBindVertexArray(obj.mesh->vao);
		glDrawElements(GL_TRIANGLES, obj.mesh->indexCount, GL_UNSIGNED_INT, 0);
	}

	glfwSwapBuffers(this->_win3D);
}

void Window3D::initialize(GLFWwindow* win2D) {
	this->_win3D = glfwCreateWindow(this->_g3DWidth, this->_g3DHeight, "3D Sandbox", nullptr, win2D);
	if (!this->_win3D) {
		std::cerr << "Failed to create 3D window.\n";
		glfwDestroyWindow(win2D);
		glfwTerminate();
		throw "Failed to create 3D window!";
	}

	glfwMakeContextCurrent(this->_win3D);
	glfwSetFramebufferSizeCallback(this->_win3D, framebuffer_size_callback_3D);
	glfwSetMouseButtonCallback(this->_win3D, mouse_button_callback_3D);
	glfwSetCursorPosCallback(this->_win3D, cursor_position_callback_3D);
	glfwSetScrollCallback(this->_win3D, scroll_callback_3D);
	glfwSetKeyCallback(this->_win3D, key_callback_3D);

	glEnable(GL_DEPTH_TEST);

	std::string vsSrcString = FileService::ReadFileContent("./Resources/3dVertShader.ver");
	std::string fsSrcString = FileService::ReadFileContent("./Resources/3dFragShader.frag");

	this->_gProg3D = ShaderService::createProgram(vsSrcString.c_str(), fsSrcString.c_str());

	// Build predefined meshes & upload
	Mesh cubeMesh = PredefinedShapeBuilderService::createBoxMesh(0.6f, 0.6f, 0.6f);
	Mesh cuboidMesh = PredefinedShapeBuilderService::createBoxMesh(1.0f, 0.6f, 0.6f);
	Mesh sphereMesh = PredefinedShapeBuilderService::createSphereMesh(0.4f, 24, 16);
	Mesh pyramidMesh = PredefinedShapeBuilderService::createPyramidMesh(0.8f, 0.8f);
	Mesh pentPrismMesh = PredefinedShapeBuilderService::createPentagonPrismMesh(0.5f, 0.8f);

	App::gCubeMesh = GPUConvertService::uploadMesh(cubeMesh);
	App::gCuboidMesh = GPUConvertService::uploadMesh(cuboidMesh);
	App::gSphereMesh = GPUConvertService::uploadMesh(sphereMesh);
	App::gPyramidMesh = GPUConvertService::uploadMesh(pyramidMesh);
	App::gPentagonPrismMesh = GPUConvertService::uploadMesh(pentPrismMesh);
}

GLFWwindow* Window3D::getWindowInstance() {
	return this->_win3D;
}

void Window3D::getRayFromMouse(double mouseX, double mouseY, int screenW, int screenH,
	const glm::mat4& projection, const glm::mat4& view,
	glm::vec3& outOrigin, glm::vec3& outDirection)
{
	float x = (2.0f * (float)mouseX) / (float)screenW - 1.0f;
	float y = 1.0f - (2.0f * (float)mouseY) / (float)screenH;
	glm::vec4 ray_clip(x, y, -1.0f, 1.0f);

	glm::vec4 ray_eye = glm::inverse(projection) * ray_clip;
	ray_eye = glm::vec4(ray_eye.x, ray_eye.y, -1.0f, 0.0f);

	glm::vec3 ray_wor = glm::vec3(glm::inverse(view) * ray_eye);
	outDirection = glm::normalize(ray_wor);
	outOrigin = glm::vec3(glm::inverse(view)[3]);
}

bool Window3D::rayIntersectsAABB(const glm::vec3& origin, const glm::vec3& dir,
	const glm::vec3& minBound, const glm::vec3& maxBound,
	float& tHit)
{
	float tMin = (minBound.x - origin.x) / dir.x;
	float tMax = (maxBound.x - origin.x) / dir.x;
	if (tMin > tMax) std::swap(tMin, tMax);

	float tyMin = (minBound.y - origin.y) / dir.y;
	float tyMax = (maxBound.y - origin.y) / dir.y;
	if (tyMin > tyMax) std::swap(tyMin, tyMax);

	if ((tMin > tyMax) || (tyMin > tMax)) return false;
	if (tyMin > tMin) tMin = tyMin;
	if (tyMax < tMax) tMax = tyMax;

	float tzMin = (minBound.z - origin.z) / dir.z;
	float tzMax = (maxBound.z - origin.z) / dir.z;
	if (tzMin > tzMax) std::swap(tzMin, tzMax);

	if ((tMin > tzMax) || (tzMin > tMax)) return false;
	if (tzMin > tMin) tMin = tzMin;
	if (tzMax < tMax) tMax = tzMax;

	tHit = tMin;
	return true;
}

glm::mat4 Window3D::getObjectModelMatrix(const SceneObject& obj) {
	glm::mat4 model(1.0f);
	model = glm::translate(model, obj.position);
	model = glm::rotate(model, glm::radians(obj.rotation.y), glm::vec3(0, 1, 0));
	model = glm::rotate(model, glm::radians(obj.rotation.x), glm::vec3(1, 0, 0));
	model = glm::scale(model, glm::vec3(obj.scale));
	return model;
}

bool Window3D::rayIntersectsObject(const SceneObject& obj,
	const glm::vec3& rayOrigin,
	const glm::vec3& rayDir,
	float& tHit)
{
	glm::mat4 model = getObjectModelMatrix(obj);
	glm::mat4 invModel = glm::inverse(model);

	glm::vec3 localOrigin = glm::vec3(invModel * glm::vec4(rayOrigin, 1.0f));
	glm::vec3 localDir = glm::normalize(glm::vec3(invModel * glm::vec4(rayDir, 0.0f)));

	return rayIntersectsAABB(localOrigin, localDir, obj.mesh->aabbMin, obj.mesh->aabbMax, tHit);
}

void Window3D::processInput3D(GLFWwindow* window) {
	if (!this->_gTransformMode && glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
		float cameraSpeed = 2.5f * this->_gDeltaTime;
		if (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS) cameraSpeed *= 2.0f;
		if (glfwGetKey(window, GLFW_KEY_W) == GLFW_PRESS) this->_gCameraPos += cameraSpeed * this->_gCameraFront;
		if (glfwGetKey(window, GLFW_KEY_S) == GLFW_PRESS) this->_gCameraPos -= cameraSpeed * this->_gCameraFront;
		if (glfwGetKey(window, GLFW_KEY_A) == GLFW_PRESS)
			this->_gCameraPos -= glm::normalize(glm::cross(this->_gCameraFront, this->_gCameraUp)) * cameraSpeed;
		if (glfwGetKey(window, GLFW_KEY_D) == GLFW_PRESS)
			this->_gCameraPos += glm::normalize(glm::cross(this->_gCameraFront, this->_gCameraUp)) * cameraSpeed;
		if (glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) this->_gCameraPos += cameraSpeed * this->_gCameraUp;
		if (glfwGetKey(window, GLFW_KEY_Q) == GLFW_PRESS) this->_gCameraPos -= cameraSpeed * this->_gCameraUp;
	}
}

void Window3D::framebuffer_size_callback_3D(GLFWwindow* window, int width, int height) {
	(void)window;
	Window3D::_g3DWidth = width;
	Window3D::_g3DHeight = height;
	glViewport(0, 0, width, height);
}

void Window3D::key_callback_3D(GLFWwindow* window, int key, int scancode, int action, int mods) {
	(void)window; (void)scancode; (void)mods;
	if (key == GLFW_KEY_T && action == GLFW_PRESS) {
		Window3D::_gTransformMode = !Window3D::_gTransformMode;
		glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		if (Window3D::_gTransformMode) std::cout << "Mode: OBJECT TRANSFORM (click objects)\n";
		else std::cout << "Mode: CAMERA\n";
	}
}

void Window3D::mouse_button_callback_3D(GLFWwindow* window, int button, int action, int mods) {
	(void)mods;

	if (Window3D::_gTransformMode) {
		if (action == GLFW_PRESS) {
			glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)Window3D::_g3DWidth / (float)Window3D::_g3DHeight, 0.1f, 100.0f);
			glm::mat4 view = glm::lookAt(Window3D::_gCameraPos, Window3D::_gCameraPos + Window3D::_gCameraFront, Window3D::_gCameraUp);

			glm::vec3 rayOrg, rayDir;
			getRayFromMouse(Window3D::_gLastX3D, Window3D::_gLastY3D, Window3D::_g3DWidth, Window3D::_g3DHeight, projection, view, rayOrg, rayDir);

			float closestDist = std::numeric_limits<float>::max();
			int hitIndex = -1;

			for (int i = 0; i < (int)App::gObjects.size(); ++i) {
				float t;
				if (rayIntersectsObject(App::gObjects[i], rayOrg, rayDir, t)) {
					if (t > 0.0f && t < closestDist) {
						closestDist = t;
						hitIndex = i;
					}
				}
			}

			if (hitIndex != -1) {
				Window3D::_gSelectedObject = hitIndex;
				Window3D::_gIsDraggingObject = true;
				if (button == GLFW_MOUSE_BUTTON_LEFT) {
					Window3D::_gLeftMouseDown = true;
					Window3D::_gDragPlanePoint = App::gObjects[hitIndex].position;
					Window3D::_gDragPlaneNormal = glm::normalize(Window3D::_gCameraFront);
					Window3D::_gHasDragPlane = true;
				}
				if (button == GLFW_MOUSE_BUTTON_RIGHT) {
					Window3D::_gRightMouseDown = true;
				}
				std::cout << "Selected object #" << Window3D::_gSelectedObject << "\n";
			}
			else {
				Window3D::_gIsDraggingObject = false;
				Window3D::_gSelectedObject = -1;
				Window3D::_gHasDragPlane = false;
			}
		}
		else if (action == GLFW_RELEASE) {
			if (button == GLFW_MOUSE_BUTTON_LEFT)  Window3D::_gLeftMouseDown = false;
			if (button == GLFW_MOUSE_BUTTON_RIGHT) Window3D::_gRightMouseDown = false;
			Window3D::_gIsDraggingObject = false;
		}
	}
	else {
		if (button == GLFW_MOUSE_BUTTON_RIGHT) {
			if (action == GLFW_PRESS)
				glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_DISABLED);
			else if (action == GLFW_RELEASE)
				glfwSetInputMode(window, GLFW_CURSOR, GLFW_CURSOR_NORMAL);
		}
	}
}

void Window3D::cursor_position_callback_3D(GLFWwindow* window, double xpos, double ypos) {
	(void)window;

	if (Window3D::_gFirstMouse3D) {
		Window3D::_gLastX3D = (float)xpos;
		Window3D::_gLastY3D = (float)ypos;
		Window3D::_gFirstMouse3D = false;
	}

	float xoffset = (float)xpos - Window3D::_gLastX3D;
	float yoffset = Window3D::_gLastY3D - (float)ypos;
	Window3D::_gLastX3D = (float)xpos;
	Window3D::_gLastY3D = (float)ypos;

	if (Window3D::_gTransformMode && Window3D::_gIsDraggingObject &&
		Window3D::_gSelectedObject >= 0 && Window3D::_gSelectedObject < (int)App::gObjects.size()) {

		SceneObject& obj = App::gObjects[Window3D::_gSelectedObject];
		float rotationSensitivity = 0.5f;

		if (Window3D::_gLeftMouseDown && Window3D::_gHasDragPlane) {
			glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)Window3D::_g3DWidth / (float)Window3D::_g3DHeight, 0.1f, 100.0f);
			glm::mat4 view = glm::lookAt(Window3D::_gCameraPos, Window3D::_gCameraPos + Window3D::_gCameraFront, Window3D::_gCameraUp);

			glm::vec3 rayOrg, rayDir;
			getRayFromMouse(Window3D::_gLastX3D, Window3D::_gLastY3D, Window3D::_g3DWidth, Window3D::_g3DHeight, projection, view, rayOrg, rayDir);

			float denom = glm::dot(rayDir, Window3D::_gDragPlaneNormal);
			if (std::fabs(denom) > 1e-6f) {
				float t = glm::dot((Window3D::_gDragPlanePoint - rayOrg), Window3D::_gDragPlaneNormal) / denom;
				if (t > 0.0f) {
					glm::vec3 hitPoint = rayOrg + t * rayDir;
					obj.position = hitPoint;
				}
			}
		}

		if (Window3D::_gRightMouseDown) {
			obj.rotation.y += xoffset * rotationSensitivity;
			obj.rotation.x += yoffset * rotationSensitivity;
		}

	}
	else if (!Window3D::_gTransformMode) {
		if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS) {
			float sensitivity = 0.1f;
			xoffset *= sensitivity;
			yoffset *= sensitivity;
			Window3D::_gYaw += xoffset;
			Window3D::_gPitch += yoffset;
			if (Window3D::_gPitch > 89.0f)  Window3D::_gPitch = 89.0f;
			if (Window3D::_gPitch < -89.0f) Window3D::_gPitch = -89.0f;

			glm::vec3 front;
			front.x = cos(glm::radians(Window3D::_gYaw)) * cos(glm::radians(Window3D::_gPitch));
			front.y = sin(glm::radians(Window3D::_gPitch));
			front.z = sin(glm::radians(Window3D::_gYaw)) * cos(glm::radians(Window3D::_gPitch));
			Window3D::_gCameraFront = glm::normalize(front);
		}
		if (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS) {
			float panSpeed = 0.05f;
			glm::vec3 cameraRight = glm::normalize(glm::cross(Window3D::_gCameraFront, Window3D::_gCameraUp));
			Window3D::_gCameraPos -= cameraRight * xoffset * panSpeed;
			Window3D::_gCameraPos -= Window3D::_gCameraUp * yoffset * panSpeed;
		}
	}
}

void Window3D::scroll_callback_3D(GLFWwindow* window, double xoffset, double yoffset) {
	(void)window; (void)xoffset;

	if (Window3D::_gTransformMode) {
		if (Window3D::_gSelectedObject >= 0 && Window3D::_gSelectedObject < (int)App::gObjects.size()) {
			SceneObject& obj = App::gObjects[Window3D::_gSelectedObject];

			glm::mat4 projection = glm::perspective(glm::radians(45.0f), (float)Window3D::_g3DWidth / (float)Window3D::_g3DHeight, 0.1f, 100.0f);
			glm::mat4 view = glm::lookAt(Window3D::_gCameraPos, Window3D::_gCameraPos + Window3D::_gCameraFront, Window3D::_gCameraUp);

			glm::vec3 rayOrg, rayDir;
			getRayFromMouse(Window3D::_gLastX3D, Window3D::_gLastY3D, Window3D::_g3DWidth, Window3D::_g3DHeight, projection, view, rayOrg, rayDir);

			float t;
			if (rayIntersectsObject(obj, rayOrg, rayDir, t)) {
				float scaleSensitivity = 0.1f;
				obj.scale += (float)yoffset * scaleSensitivity;
				if (obj.scale < 0.1f) obj.scale = 0.1f;
			}
		}
	}
	else {
		float zoomSpeed = 1.0f;
		Window3D::_gCameraPos += Window3D::_gCameraFront * (float)yoffset * zoomSpeed;
	}
}