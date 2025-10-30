#include <iostream>
#include <opencv2/opencv.hpp>
#include "Window.h"
#include "Shader.h"
#include "../Services/TransformService.h"
#include "../Models/Filters.h"

#define INIT_SUCCESS 0

// public methods

Window::Window(int initialW, int inititalH) {
	this->_initialW = initialW;
	this->_initialH = inititalH;
	this->_win = glfwCreateWindow(initialW, _initialH, "Assignment 2", nullptr, nullptr);
	this->_logger = new LoggerService();
	this->_cap = cv::VideoCapture(0, cv::CAP_ANY);
	this->_frameBGR = cv::Mat::zeros(0, 0, CV_8UC3);
	this->_camW = 0;
	this->_camH = 0;
	this->_tex = 0;
	this->_vao = 0;
	this->_vbo = 0;
	this->_ebo = 0;
	this->_uTransformLoc = -1;
	this->_uTexLoc = -1;
	this->_uScaleLoc = -1;
	this->_uFilterTypeLoc = -1;
	this->_uPixelBlockLoc = -1;
	this->_uRedThreshLoc = -1;
	this->_prog = 0;
	this->_useGPU = true;
	this->_filter = F_NONE;
	this->_pixelBlock = 12;
	this->_redThresh = 0.6f;
	this->_cpuOutBGR = cv::Mat::zeros(0, 0, CV_8UC3);
	this->_state = AppState();
}

Window::~Window() {
	delete _logger;

	glDeleteProgram(this->_prog);
	glDeleteBuffers(1, &this->_ebo);
	glDeleteBuffers(1, &this->_vbo);
	glDeleteVertexArrays(1, &this->_vao);
	glDeleteTextures(1, &this->_tex);

	glfwDestroyWindow(_win);
	glfwTerminate();
}

int Window::init() {
	try
	{
		if (!this->_win) {
			glfwTerminate();
			throw std::runtime_error("Failed to create GLFW window");
		}

		glfwMakeContextCurrent(_win);

		if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
			throw std::runtime_error("Failed to initialize GLAD");
		}

		glfwSwapInterval(1); // vsync

		this->_logger->LogControls();

		if (!this->_cap.isOpened()) {
			throw std::runtime_error("Failed to open default camera");
		}
		this->_cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
		this->_cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
		this->_cap.set(cv::CAP_PROP_CONVERT_RGB, true);

		// Read one frame to learn width/height
		cv::Mat frameBGR;
		if (!this->_cap.read(frameBGR) || frameBGR.empty()) {
			throw std::runtime_error("Could not read from camera");
		}

		this->_camW = frameBGR.cols;
		this->_camH = frameBGR.rows;

		_logger->LogCameraResolution(this->_camW, this->_camH);

		// Create GL texture
		glGenTextures(1, &this->_tex);
		glBindTexture(GL_TEXTURE_2D, this->_tex);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, this->_camW, this->_camH, 0, GL_BGR, GL_UNSIGNED_BYTE, frameBGR.data);

		// Fullscreen quad
		const float quad[] = {
			-1.f, -1.f, 0.f, 0.f,
			 1.f, -1.f, 1.f, 0.f,
			 1.f,  1.f, 1.f, 1.f,
			-1.f,  1.f, 0.f, 1.f
		};
		const unsigned int idx[] = { 0,1,2, 2,3,0 };

		glGenVertexArrays(1, &this->_vao);
		glBindVertexArray(this->_vao);
		glGenBuffers(1, &this->_vbo);
		glBindBuffer(GL_ARRAY_BUFFER, this->_vbo);
		glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
		glGenBuffers(1, &this->_ebo);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, this->_ebo);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);

		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

		// Program and uniforms
		this->_prog = Shader::makeProgram(this->_uTransformLoc, this->_uTexLoc, this->_uScaleLoc, this->_uFilterTypeLoc, this->_uPixelBlockLoc, this->_uRedThreshLoc);

		if (!this->_prog) {
			throw std::runtime_error("Shader program creation failed");
		}

		this->_state.camW = this->_camW;
		this->_state.camH = this->_camH;

		glfwSetWindowUserPointer(_win, &this->_state);
		glfwSetCursorPosCallback(_win, this->cursorPosCB);
		glfwSetMouseButtonCallback(_win, this->mouseButtonCB);
		glfwSetScrollCallback(_win, this->scrollCB);
	}
	catch (const std::exception& e)
	{
		std::cerr << e.what() << std::endl;
		return EXIT_FAILURE;
	}

	return INIT_SUCCESS;
}

void Window::run() {
	auto t0 = std::chrono::steady_clock::now();
	int frames = 0;

	// Main loop for frame update
	while (!glfwWindowShouldClose(_win)) {
		// Poll input
		glfwPollEvents();

		if (glfwGetKey(_win, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(_win, GLFW_TRUE);
		if (glfwGetKey(_win, GLFW_KEY_1) == GLFW_PRESS) this->_filter = F_NONE;
		if (glfwGetKey(_win, GLFW_KEY_2) == GLFW_PRESS) this->_filter = F_PIXELATE;
		if (glfwGetKey(_win, GLFW_KEY_3) == GLFW_PRESS) this->_filter = F_SINCITY;
		if (glfwGetKey(_win, GLFW_KEY_4) == GLFW_PRESS) this->_filter = F_COMIC;
		if (glfwGetKey(_win, GLFW_KEY_G) == GLFW_PRESS) this->_useGPU = true;
		if (glfwGetKey(_win, GLFW_KEY_C) == GLFW_PRESS) this->_useGPU = false;
		if (glfwGetKey(_win, GLFW_KEY_UP) == GLFW_PRESS) this->_pixelBlock = std::min(256, this->_pixelBlock + 1);
		if (glfwGetKey(_win, GLFW_KEY_DOWN) == GLFW_PRESS) this->_pixelBlock = std::max(1, this->_pixelBlock - 1);
		if (glfwGetKey(_win, GLFW_KEY_R) == GLFW_PRESS) { this->_state.rot = 0.0f; this->_state.scale = 1.0f; this->_state.tx = 0.0f; this->_state.ty = 0.0f; }

		// Grab a frame (BGR8)
		if (!this->_cap.read(this->_frameBGR) || this->_frameBGR.empty()) {
			continue;
		}

		// CPU path: warp first with same transform, then apply CPU filter
		if (!this->_useGPU) {

			cv::Mat warped;
			cv::Mat Ainv;

			TransformService::uvForward_to_cvInverse2x3(this->_state, this->_camW, this->_camH, Ainv);

			cv::warpAffine(this->_frameBGR, warped, Ainv, cv::Size(this->_camW, this->_camH),
				cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

			switch (this->_filter) {
			case F_PIXELATE: FilterModel::pixelateCPU(warped, this->_cpuOutBGR, this->_pixelBlock); break;
			case F_SINCITY:  FilterModel::sinCityCPU(warped, this->_cpuOutBGR, 0); break;
			case F_COMIC:    FilterModel::comicCPU(warped, this->_cpuOutBGR); break;
			default:         this->_cpuOutBGR = warped; break;
			}

			glBindTexture(GL_TEXTURE_2D, this->_tex);
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->_camW, this->_camH, GL_BGR, GL_UNSIGNED_BYTE, this->_cpuOutBGR.data);
		}
		else {
			// GPU branch: upload raw camera frame, shader will apply transform and effect
			glBindTexture(GL_TEXTURE_2D, this->_tex);
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->_camW, this->_camH, GL_BGR, GL_UNSIGNED_BYTE, this->_frameBGR.data);
		}

		// Compute aspect-preserving scale for letterboxing
		int winW;
		int winH;

		glfwGetFramebufferSize(_win, &winW, &winH);

		float winAspect = (float)winW / std::max(1, winH);
		float camAspect = (float)this->_camW / (float)this->_camH;
		float sx = 1.f, sy = 1.f;

		if (winAspect > camAspect) {
			sx = camAspect / winAspect;
		}
		else {
			sy = winAspect / camAspect;
		}

		// Stash for mouse mapping
		this->_state.winW = winW;
		this->_state.winH = winH;
		this->_state.sx = sx;
		this->_state.sy = sy;

		// Render
		glViewport(0, 0, winW, winH);
		glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		glUseProgram(this->_prog);
		glUniform1i(this->_uTexLoc, 0);
		glUniform2f(this->_uScaleLoc, sx, sy);
		glUniform1i(this->_uFilterTypeLoc, this->_useGPU ? this->_filter : 0); // 0 in CPU mode
		glUniform1f(this->_uPixelBlockLoc, (float)this->_pixelBlock);
		glUniform1f(this->_uRedThreshLoc, this->_redThresh);

		// Upload transform matrix (identity in CPU mode to avoid double-transform)
		if (!this->_useGPU) {
			GLfloat I[9] = { 1,0,0, 0,1,0, 0,0,1 };
			glUniformMatrix3fv(this->_uTransformLoc, 1, GL_FALSE, I);
		}
		else {
			Mat3 Muv = TransformService::buildUvForward(this->_state);
			GLfloat matData[9] = {
				Muv.a11, Muv.a21, Muv.a31,
				Muv.a12, Muv.a22, Muv.a32,
				Muv.a13, Muv.a23, Muv.a33
			};
			glUniformMatrix3fv(this->_uTransformLoc, 1, GL_FALSE, matData);
		}

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, this->_tex);

		glBindVertexArray(this->_vao);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

		glfwSwapBuffers(_win);

		// FPS meter
		frames++;

		auto t1 = std::chrono::steady_clock::now();
		if (std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count() >= 1) {
			
			_logger->LogStatusOfApp(frames, this->_pixelBlock, this->_filter, this->_useGPU, this->_state);
			
			frames = 0; t0 = t1;
		}
	}
}

void Window::cursorPosCB(GLFWwindow* w, double x, double y) {
	auto* st = (AppState*)glfwGetWindowUserPointer(w);
	if (!st) return;
	double dx = x - st->lastX;
	double dy = y - st->lastY;
	st->lastX = x; st->lastY = y;

	if (st->draggingPan) {
		// Map window pixels to UV motion, accounting for letterbox scale
		st->tx -= float(dx) / (double(st->winW) * st->sx);
		st->ty -= float(dy) / (double(st->winH) * st->sy);
		st->tx = std::max(-2.0f, std::min(2.0f, st->tx));
		st->ty = std::max(-2.0f, std::min(2.0f, st->ty));
	}
	if (st->draggingRot) {
		st->rot += float(dx) * 0.005f; // horizontal drag rotates
	}
}

void Window::mouseButtonCB(GLFWwindow* w, int button, int action, int mods) {
	auto* st = (AppState*)glfwGetWindowUserPointer(w);
	if (!st) return;
	if (action == GLFW_PRESS) {
		glfwGetCursorPos(w, &st->lastX, &st->lastY);
		if (button == GLFW_MOUSE_BUTTON_LEFT && !(mods & GLFW_MOD_SHIFT))
			st->draggingPan = true;
		if (button == GLFW_MOUSE_BUTTON_RIGHT || (button == GLFW_MOUSE_BUTTON_LEFT && (mods & GLFW_MOD_SHIFT)))
			st->draggingRot = true;
	}
	else if (action == GLFW_RELEASE) {
		if (button == GLFW_MOUSE_BUTTON_LEFT) { st->draggingPan = false; st->draggingRot = false; }
		if (button == GLFW_MOUSE_BUTTON_RIGHT) { st->draggingRot = false; }
	}
}

void Window::scrollCB(GLFWwindow* w, double /*xoff*/, double yoff) {
	auto* st = (AppState*)glfwGetWindowUserPointer(w);
	if (!st) return;
	float k = (yoff > 0) ? 1.1f : 0.9f;
	st->scale /= k;
	st->scale = std::max(0.05f, std::min(10.0f, st->scale));
}

// private methods
