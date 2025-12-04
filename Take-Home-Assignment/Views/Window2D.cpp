#include "window2D.h"
#include <cmath>
#include <vector>
#include "../Models/Constants.h"
#include "../Models/DetectedShape.h"
#include "../Models/GLMesh.h"
#include "../Models/Mesh.h"
#include "../Services/ShapeFormationService.h"
#include "../Services/GPUConvertService.h"
#include "../Models/Constants.h"
#include "../Services/FileService.h"
#include "../Services/ShaderService.h"
#include "../App.h"

Window2D::Window2D() {
	this->_win2D = nullptr;
	this->_gProg2D = 0;
}

void Window2D::run() {
	glfwMakeContextCurrent(this->_win2D);
	int fbw2, fbh2;
	glfwGetFramebufferSize(this->_win2D, &fbw2, &fbh2);
	glViewport(0, 0, fbw2, fbh2);

	if (App::g_state.needUpdate) {
		cv::Mat display = App::g_state.canvas.clone();
		cv::line(display, cv::Point(AppConstants::DRAW_W / 2, 0), cv::Point(AppConstants::DRAW_W / 2, AppConstants::DRAW_H), cv::Scalar(200, 200, 200), 1);
		cv::line(display, cv::Point(0, AppConstants::DRAW_H / 2), cv::Point(AppConstants::DRAW_W, AppConstants::DRAW_H / 2), cv::Scalar(200, 200, 200), 1);

		cv::Mat flipped;
		cv::flip(display, flipped, 0);

		glBindTexture(GL_TEXTURE_2D, App::g_state.tex);
		glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, AppConstants::DRAW_W, AppConstants::DRAW_H,
			GL_BGR, GL_UNSIGNED_BYTE, flipped.data);

		App::g_state.needUpdate = false;
	}

	glClearColor(0.2f, 0.25f, 0.3f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT);

	glUseProgram(this->_gProg2D);
	glUniform1i(glGetUniformLocation(this->_gProg2D, "uTex"), 0);
	glActiveTexture(GL_TEXTURE0);
	glBindTexture(GL_TEXTURE_2D, App::g_state.tex);

	glBindVertexArray(App::g_state.quadVAO);
	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

	glfwSwapBuffers(this->_win2D);
}

GLFWwindow* Window2D::getWindowInstance() {
	return this->_win2D;
}

void Window2D::initialize(GLFWwindow* _) {
	this->_win2D = glfwCreateWindow(AppConstants::DRAW_W, AppConstants::DRAW_H, "2D Drawing", nullptr, nullptr);
	if (!this->_win2D) {
		std::cerr << "Failed to create 2D window.\n";
		glfwTerminate();
		throw "Failed to create 2D window!";
	}

	glfwMakeContextCurrent(this->_win2D);
	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cerr << "Failed to init GLAD.\n";
		glfwTerminate();
		throw "Failed to init GLAD.";
	}

	App::g_state.canvas = cv::Mat(AppConstants::DRAW_H, AppConstants::DRAW_W, CV_8UC3, cv::Scalar(255, 255, 255));

	glfwSetMouseButtonCallback(this->_win2D, mouse_button_callback);
	glfwSetCursorPosCallback(this->_win2D, cursor_position_callback);
	glfwSetKeyCallback(this->_win2D, key_callback_2D);

	std::string vsSrcString = FileService::ReadFileContent("./Resources/2dVertShader.ver");
	std::string fsSrcString = FileService::ReadFileContent("./Resources/2dFragShader.frag");

	this->_gProg2D = ShaderService::createProgram(vsSrcString.c_str(), fsSrcString.c_str());

	float quadVerts[] = {
		// pos      // tex
		-1.0f, -1.0f, 0.0f, 0.0f,
		 1.0f, -1.0f, 1.0f, 0.0f,
		 1.0f,  1.0f, 1.0f, 1.0f,
		-1.0f,  1.0f, 0.0f, 1.0f
	};

	unsigned int quadIdx[] = { 0,1,2, 2,3,0 };

	glGenVertexArrays(1, &App::g_state.quadVAO);
	glGenBuffers(1, &App::g_state.quadVBO);
	glGenBuffers(1, &App::g_state.quadEBO);

	glBindVertexArray(App::g_state.quadVAO);
	glBindBuffer(GL_ARRAY_BUFFER, App::g_state.quadVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts), quadVerts, GL_STATIC_DRAW);

	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, App::g_state.quadEBO);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(quadIdx), quadIdx, GL_STATIC_DRAW);

	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
	glEnableVertexAttribArray(1);

	// texture for canvas
	glGenTextures(1, &App::g_state.tex);
	glBindTexture(GL_TEXTURE_2D, App::g_state.tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, AppConstants::DRAW_W, AppConstants::DRAW_H, 0,
		GL_BGR, GL_UNSIGNED_BYTE, App::g_state.canvas.data);
}

void Window2D::mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
	(void)window; (void)mods;
	if (button == GLFW_MOUSE_BUTTON_LEFT) {
		if (action == GLFW_PRESS) {
			App::g_state.drawing = true;
			glfwGetCursorPos(window, &App::g_state.lastX, &App::g_state.lastY);
		}
		else if (action == GLFW_RELEASE) {
			App::g_state.drawing = false;
		}
	}
}

void Window2D::cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
	(void)window;
	if (!App::g_state.drawing) return;

	double x = std::max(0.0, std::min(xpos, (double)AppConstants::DRAW_W - 1));
	double y = std::max(0.0, std::min(ypos, (double)AppConstants::DRAW_H - 1));

	cv::Point p1((int)App::g_state.lastX, (int)App::g_state.lastY);
	cv::Point p2((int)x, (int)y);
	cv::line(App::g_state.canvas, p1, p2, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);

	App::g_state.lastX = x;
	App::g_state.lastY = y;
	App::g_state.needUpdate = true;
}

glm::vec3 Window2D::randomBrightColor() {
	auto r = []() { return 0.5f + 0.5f * (float)rand() / (float)RAND_MAX; };
	return glm::vec3(r(), r(), r());
}

void Window2D::key_callback_2D(GLFWwindow* window, int key, int scancode, int action, int mods) {
	(void)window; (void)scancode; (void)mods;
	if (action != GLFW_PRESS) return;

	// C — clear
	if (key == GLFW_KEY_C) {
		App::g_state.canvas.setTo(cv::Scalar(255, 255, 255));
		App::g_state.needUpdate = true;
		std::cout << "Canvas cleared.\n";
		return;
	}

	// R — predefined shapes
	if (key == GLFW_KEY_R) {
		DetectedShape ds = ShapeFormationService::detectShapeWithPolygon(App::g_state.canvas);
		std::cout << "Recognized: " << ds.label << "\n";

		GLMesh* chosen = nullptr;
		if (ds.label == "circle")        chosen = &App::gSphereMesh;
		else if (ds.label == "square")   chosen = &App::gCubeMesh;
		else if (ds.label == "rectangle")chosen = &App::gCuboidMesh;
		else if (ds.label == "triangle") chosen = &App::gPyramidMesh;
		else if (ds.label == "pentagon") chosen = &App::gPentagonPrismMesh;
		else {
			std::cout << "No predefined 3D shape.\n";
			return;
		}

		int idx = (int)App::gObjects.size();
		int row = idx / 5;
		int col = idx % 5;
		float tx = (col - 2) * 0.8f;
		float tz = -row * 0.8f;

		SceneObject obj;
		obj.mesh = chosen;
		obj.position = glm::vec3(tx, 0.0f, tz);
		obj.rotation = glm::vec3(0.0f);
		obj.scale = 1.0f;
		obj.color = randomBrightColor();

		App::gObjects.push_back(obj);

		std::cout << "Added predefined 3D shape.\n";
		return;
	}

	// X — extrusion
	if (key == GLFW_KEY_X) {
		std::vector<cv::Point> raw = ShapeFormationService::getRawDrawnStroke(App::g_state.canvas);
		if (raw.empty()) {
			std::cout << "No raw stroke to revolve.\n";
			return;
		}

		Mesh m = ShapeFormationService::extrudeY(raw, 0.6f);
		GLMesh gm = GPUConvertService::uploadMesh(m);
		if (!gm.vao) {
			std::cout << "Failed to upload extruded mesh.\n";
			return;
		}

		App::gDynamicMeshes.push_back(gm);
		GLMesh* ptr = &App::gDynamicMeshes.back();

		int idx = (int)App::gObjects.size();
		int row = idx / 5;
		int col = idx % 5;
		float tx = (col - 2) * 0.8f;
		float tz = -row * 0.8f;

		SceneObject obj;
		obj.mesh = ptr;
		obj.position = glm::vec3(tx, 0.0f, tz);
		obj.rotation = glm::vec3(0.0f);
		obj.scale = 1.0f;
		obj.color = randomBrightColor();

		App::gObjects.push_back(obj);

		std::cout << "Added extruded shape.\n";
		return;
	}

	// V — revolve RAW stroke around X-axis
	if (key == GLFW_KEY_V) {
		std::vector<cv::Point> raw = ShapeFormationService::getRawDrawnStroke(App::g_state.canvas);
		if (raw.empty()) {
			std::cout << "No raw stroke to revolve.\n";
			return;
		}

		Mesh m = ShapeFormationService::revolvePolygon(raw, 'x', 64);
		GLMesh gm = GPUConvertService::uploadMesh(m);
		if (!gm.vao) {
			std::cout << "Failed to upload revolution mesh.\n";
			return;
		}

		App::gDynamicMeshes.push_back(gm);
		GLMesh* ptr = &App::gDynamicMeshes.back();

		int idx = (int)App::gObjects.size();
		int row = idx / 5;
		int col = idx % 5;
		float tx = (col - 2) * 0.8f;
		float tz = -row * 0.8f;

		SceneObject obj;
		obj.mesh = ptr;
		obj.position = glm::vec3(tx, 0.0f, tz);
		obj.rotation = glm::vec3(0.0f);
		obj.scale = 1.0f;
		obj.color = randomBrightColor();

		App::gObjects.push_back(obj);

		std::cout << "Added revolution (X-axis) shape.\n";
		return;
	}

	// H — revolve RAW stroke around Y-axis
	if (key == GLFW_KEY_H) {
		std::vector<cv::Point> raw = ShapeFormationService::getRawDrawnStroke(App::g_state.canvas);
		if (raw.empty()) {
			std::cout << "No raw stroke to revolve.\n";
			return;
		}

		Mesh m = ShapeFormationService::revolvePolygon(raw, 'y', 64);
		GLMesh gm = GPUConvertService::uploadMesh(m);
		if (!gm.vao) {
			std::cout << "Failed to upload revolution mesh.\n";
			return;
		}

		App::gDynamicMeshes.push_back(gm);
		GLMesh* ptr = &App::gDynamicMeshes.back();

		int idx = (int)App::gObjects.size();
		int row = idx / 5;
		int col = idx % 5;
		float tx = (col - 2) * 0.8f;
		float tz = -row * 0.8f;

		SceneObject obj;
		obj.mesh = ptr;
		obj.position = glm::vec3(tx, 0.0f, tz);
		obj.rotation = glm::vec3(0.0f);
		obj.scale = 1.0f;
		obj.color = randomBrightColor();

		App::gObjects.push_back(obj);

		std::cout << "Added revolution (Y-axis) shape.\n";
		return;
	}
}

