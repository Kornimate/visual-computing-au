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

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#include <iostream>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <utility>

#include <glad/glad.h>
#include <GLFW/glfw3.h>

#include <opencv2/opencv.hpp>

// === Utility: Check & print shader compile / link errors ===
static bool checkShader(GLuint obj, bool isProgram, const char* label) {
	GLint status = GL_FALSE; GLint logLen = 0;
	if (isProgram) {
		glGetProgramiv(obj, GL_LINK_STATUS, &status);
		glGetProgramiv(obj, GL_INFO_LOG_LENGTH, &logLen);
	}
	else {
		glGetShaderiv(obj, GL_COMPILE_STATUS, &status);
		glGetShaderiv(obj, GL_INFO_LOG_LENGTH, &logLen);
	}
	if (status == GL_FALSE) {
		std::vector<char> log(std::max(1, logLen));
		if (isProgram) glGetProgramInfoLog(obj, logLen, nullptr, log.data());
		else glGetShaderInfoLog(obj, logLen, nullptr, log.data());
		std::cerr << "[GL] " << label << " error:\n" << log.data() << std::endl;
		return false;
	}
	return true;
}

// === Small 3x3 matrix helper for UV-space affine ===
struct Mat3 {
	float a11 = 1, a12 = 0, a13 = 0;
	float a21 = 0, a22 = 1, a23 = 0;
	float a31 = 0, a32 = 0, a33 = 1;
};
static Mat3 matMul(const Mat3& A, const Mat3& B) { // matrix computation 
	Mat3 M;
	M.a11 = A.a11 * B.a11 + A.a12 * B.a21 + A.a13 * B.a31;
	M.a12 = A.a11 * B.a12 + A.a12 * B.a22 + A.a13 * B.a32;
	M.a13 = A.a11 * B.a13 + A.a12 * B.a23 + A.a13 * B.a33;
	M.a21 = A.a21 * B.a11 + A.a22 * B.a21 + A.a23 * B.a31;
	M.a22 = A.a21 * B.a12 + A.a22 * B.a22 + A.a23 * B.a32;
	M.a23 = A.a21 * B.a13 + A.a22 * B.a23 + A.a23 * B.a33;
	M.a31 = A.a31 * B.a11 + A.a32 * B.a21 + A.a33 * B.a31;
	M.a32 = A.a31 * B.a12 + A.a32 * B.a22 + A.a33 * B.a32;
	M.a33 = A.a31 * B.a13 + A.a32 * B.a23 + A.a33 * B.a33;
	return M;
}
static Mat3 matInv(const Mat3& m) {
	Mat3 r;
	float d =
		m.a11 * (m.a22 * m.a33 - m.a23 * m.a32) -
		m.a12 * (m.a21 * m.a33 - m.a23 * m.a31) +
		m.a13 * (m.a21 * m.a32 - m.a22 * m.a31);
	if (std::fabs(d) < 1e-8f) return Mat3(); // identity fallback
	float inv = 1.0f / d;
	r.a11 = (m.a22 * m.a33 - m.a23 * m.a32) * inv;
	r.a12 = -(m.a12 * m.a33 - m.a13 * m.a32) * inv;
	r.a13 = (m.a12 * m.a23 - m.a13 * m.a22) * inv;
	r.a21 = -(m.a21 * m.a33 - m.a23 * m.a31) * inv;
	r.a22 = (m.a11 * m.a33 - m.a13 * m.a31) * inv;
	r.a23 = -(m.a11 * m.a23 - m.a13 * m.a21) * inv;
	r.a31 = (m.a21 * m.a32 - m.a22 * m.a31) * inv;
	r.a32 = -(m.a11 * m.a32 - m.a12 * m.a31) * inv;
	r.a33 = (m.a11 * m.a22 - m.a12 * m.a21) * inv;
	return r;
}
static Mat3 T(float tx, float ty) { Mat3 m; m.a13 = tx; m.a23 = ty; return m; }
static Mat3 S(float s) { Mat3 m; m.a11 = s; m.a22 = s; return m; }
static Mat3 R(float ang) {
	Mat3 m; float c = std::cos(ang), s = std::sin(ang);
	m.a11 = c; m.a12 = -s; m.a21 = s; m.a22 = c; return m;
}

// === Interactive App State ===
struct AppState {
	// Transform in UV space (u,v in [0,1])
	float rot = 0.0f;     // radians
	float scale = 1.0f;   // uniform
	float tx = 0.0f;      // translation (UV units)
	float ty = 0.0f;

	bool draggingPan = false, draggingRot = false;
	double lastX = 0.0, lastY = 0.0;

	// framebuffer size & letterbox scale for correct mouse mapping
	int winW = 1280, winH = 720;
	float sx = 1.f, sy = 1.f;

	// camera size
	int camW = 1, camH = 1;
};

static Mat3 buildUvForward(const AppState& st) {
	// Center about (0.5,0.5), apply S, R, then T in UV space
	Mat3 M = T(st.tx, st.ty);
	M = matMul(M, T(0.5f, 0.5f));
	M = matMul(M, R(st.rot));
	M = matMul(M, S(st.scale));
	M = matMul(M, T(-0.5f, -0.5f));
	return M;
}

// Callbacks
static void cursorPosCB(GLFWwindow* w, double x, double y) {
	auto* st = (AppState*)glfwGetWindowUserPointer(w);
	if (!st) return;
	double dx = x - st->lastX;
	double dy = y - st->lastY;
	st->lastX = x; st->lastY = y;

	if (st->draggingPan) {
		// Map window pixels to UV motion, accounting for letterbox scale
		st->tx += float(dx) / (double(st->winW) * st->sx);
		st->ty -= float(dy) / (double(st->winH) * st->sy);
		st->tx = std::max(-2.0f, std::min(2.0f, st->tx));
		st->ty = std::max(-2.0f, std::min(2.0f, st->ty));
	}
	if (st->draggingRot) {
		st->rot += float(dx) * 0.005f; // horizontal drag rotates
	}
}
static void mouseButtonCB(GLFWwindow* w, int button, int action, int mods) {
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
static void scrollCB(GLFWwindow* w, double /*xoff*/, double yoff) {
	auto* st = (AppState*)glfwGetWindowUserPointer(w);
	if (!st) return;
	float k = (yoff > 0) ? 1.1f : 0.9f;
	st->scale *= k;
	st->scale = std::max(0.05f, std::min(10.0f, st->scale));
}

// Convert forward UV transform to OpenCV inverse 2x3 in pixel coords
static void uvForward_to_cvInverse2x3(const AppState& st, int camW, int camH, cv::Mat& cv2x3Inv)
{
	// UV<->PX scale matrices
	Mat3 Suv2px; Suv2px.a11 = (float)camW; Suv2px.a22 = (float)camH;
	Mat3 Spx2uv; Spx2uv.a11 = 1.0f / (float)camW; Spx2uv.a22 = 1.0f / (float)camH;

	Mat3 Fuv = buildUvForward(st);               // src(UV) -> dst(UV)
	Mat3 Fpx = matMul(Suv2px, matMul(Fuv, Spx2uv)); // src(px) -> dst(px)
	Mat3 Finv = matInv(Fpx);                     // dst(px) -> src(px)

	cv2x3Inv = (cv::Mat_<float>(2, 3) <<
		Finv.a11, Finv.a12, Finv.a13,
		Finv.a21, Finv.a22, Finv.a23);
}

// === GLSL Program (supports GPU filters) ===
static GLuint makeProgram(GLint& uTransformLocOut, GLint& uTexLocOut, GLint& uScaleLocOut,
	GLint& uFilterTypeLocOut, GLint& uPixelBlockLocOut, GLint& uRedThreshLocOut)
{
	const char* vsSrc = R"GLSL(
        #version 330 core
        layout(location=0) in vec2 aPos;      // NDC quad
        layout(location=1) in vec2 aUV;       // tex coords
        out vec2 vUV;
        uniform vec2 uScale;  // aspect-correct scale for letterboxing
        void main(){
            vec2 p = aPos * uScale;          // scale to preserve aspect
            gl_Position = vec4(p, 0.0, 1.0);
            vUV = vec2(aUV.x, 1.0 - aUV.y);  // flip V so no CPU flip needed
        }
    )GLSL";

	const char* fsSrc = R"GLSL(
        #version 330 core
        in vec2 vUV;
        out vec4 frag;
        uniform sampler2D uTex;
        uniform int   uFilterType;   // 0=none,1=pixelate,2=sincity,3=comic
        uniform float uPixelBlock;   // block size for pixelation
        uniform float uRedThreshold; // sincity red threshold (0..1)
        uniform mat3  uTransform;    // forward UV transform

        bool in01(vec2 uv){ return all(greaterThanEqual(uv, vec2(0.0))) &&
                                   all(lessThanEqual(uv, vec2(1.0))); }

        vec3 applyPixelate(vec2 uv){
            float block = max(1.0, uPixelBlock);
            ivec2 ts = textureSize(uTex, 0);
            vec2 texel = 1.0 / vec2(ts);
            vec2 grid = floor(uv * vec2(ts) / block) * block * texel;
            return texture(uTex, grid).rgb;
        }
        vec3 applySinCity(vec2 uv){
            vec3 c = texture(uTex, uv).rgb;
            float gray = dot(c, vec3(0.299, 0.587, 0.114));
            float maxGB = max(c.g, c.b);
            bool isRed = (c.r > uRedThreshold) && (c.r - maxGB > 0.2);
            return isRed ? c : vec3(gray);
        }
        vec3 applyComic(vec2 uv){
            vec3 c = texture(uTex,uv).rgb;
            float edge = length(fwidth(c));
            float ink = step(0.15, edge);
            float q = 5.0;
            vec3 cq = floor(c*q)/q;
            return mix(vec3(0.0), cq, 1.0-ink);
        }

        void main(){
            vec2 uv = (uTransform * vec3(vUV,1.0)).xy;
            if(!in01(uv)){ frag = vec4(0.0,0.0,0.0,1.0); return; }

            vec3 col;
            if(uFilterType==1){
                col = applyPixelate(uv);
            } else if(uFilterType==2){
                col = applySinCity(uv);
            } else if(uFilterType==3){
                col = applyComic(uv);
            } else {
                col = texture(uTex, uv).rgb;
            }
            frag = vec4(col, 1.0);
        }
    )GLSL";

	GLuint vs = glCreateShader(GL_VERTEX_SHADER);
	glShaderSource(vs, 1, &vsSrc, nullptr);
	glCompileShader(vs);
	if (!checkShader(vs, false, "Vertex Shader")) return 0;

	GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
	glShaderSource(fs, 1, &fsSrc, nullptr);
	glCompileShader(fs);
	if (!checkShader(fs, false, "Fragment Shader")) return 0;

	GLuint prog = glCreateProgram();
	glAttachShader(prog, vs);
	glAttachShader(prog, fs);
	glLinkProgram(prog);
	if (!checkShader(prog, true, "Program")) return 0;

	glDeleteShader(vs);
	glDeleteShader(fs);

	// Uniform locations
	uTransformLocOut = glGetUniformLocation(prog, "uTransform");
	uTexLocOut = glGetUniformLocation(prog, "uTex");
	uScaleLocOut = glGetUniformLocation(prog, "uScale");
	uFilterTypeLocOut = glGetUniformLocation(prog, "uFilterType");
	uPixelBlockLocOut = glGetUniformLocation(prog, "uPixelBlock");
	uRedThreshLocOut = glGetUniformLocation(prog, "uRedThreshold");

	return prog;
}

// === CPU Filters (OpenCV) ===
static void pixelateCPU(const cv::Mat& srcBGR, cv::Mat& dstBGR, int block) {
	block = std::max(1, block);
	int smallW = std::max(1, srcBGR.cols / block);
	int smallH = std::max(1, srcBGR.rows / block);
	cv::resize(srcBGR, dstBGR, cv::Size(smallW, smallH), 0, 0, cv::INTER_NEAREST);
	cv::resize(dstBGR, dstBGR, srcBGR.size(), 0, 0, cv::INTER_NEAREST);
}
static void sinCityCPU(const cv::Mat& srcBGR, cv::Mat& dstBGR, int /*unused*/, int preserveHue = 0) {
	(void)preserveHue;
	// Keep REDs, grayscale others.
	cv::Mat hsv, gray, mask1, mask2, mask;
	cv::cvtColor(srcBGR, hsv, cv::COLOR_BGR2HSV);
	// Red wraps around hue (0..179 in OpenCV)
	cv::inRange(hsv, cv::Scalar(0, 80, 60), cv::Scalar(10, 255, 255), mask1);
	cv::inRange(hsv, cv::Scalar(170, 80, 60), cv::Scalar(180, 255, 255), mask2);
	cv::bitwise_or(mask1, mask2, mask);

	cv::cvtColor(srcBGR, gray, cv::COLOR_BGR2GRAY);
	cv::cvtColor(gray, dstBGR, cv::COLOR_GRAY2BGR);
	srcBGR.copyTo(dstBGR, mask); // copy colored pixels where mask is true
}
static void comicCPU(const cv::Mat& srcBGR, cv::Mat& dstBGR) {
	cv::Mat gray, edges;
	cv::cvtColor(srcBGR, gray, cv::COLOR_BGR2GRAY);
	cv::medianBlur(gray, gray, 7);
	cv::Laplacian(gray, edges, CV_8U, 5);
	cv::threshold(edges, edges, 80, 255, cv::THRESH_BINARY_INV);
	cv::Mat quant;
	cv::pyrMeanShiftFiltering(srcBGR, quant, 8, 16);
	cv::bitwise_and(quant, quant, dstBGR, edges);
}

int main() {
	// --- Init GLFW ---
	if (!glfwInit()) {
		std::cerr << "Failed to init GLFW" << std::endl;
		return EXIT_FAILURE;
	}
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif

	const int initialW = 1280, initialH = 720;
	GLFWwindow* win = glfwCreateWindow(initialW, initialH, "Webcam → OpenGL (CPU/GPU Filters + Affine Transform)", nullptr, nullptr);
	if (!win) {
		std::cerr << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return EXIT_FAILURE;
	}
	glfwMakeContextCurrent(win);

	if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
		std::cerr << "Failed to initialize GLAD" << std::endl;
		return EXIT_FAILURE;
	}
	glfwSwapInterval(1); // vsync

	std::cout << "GL: " << glGetString(GL_VERSION) << "\n";
	std::cout << "Controls: [1] Pixelate  [2] SinCity [3] Comic [0] None  |  [G] GPU  [C] CPU"
		<< "  |  Left-drag: pan  Right-drag/Shift+L: rotate  Wheel: zoom  [R]: reset  [Esc]: Quit\n";

	// --- Init OpenCV capture ---
	cv::VideoCapture cap(0, cv::CAP_ANY);
	if (!cap.isOpened()) {
		std::cerr << "Failed to open default camera" << std::endl;
		return EXIT_FAILURE;
	}
	cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
	cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
	cap.set(cv::CAP_PROP_CONVERT_RGB, true);

	// Read one frame to learn width/height
	cv::Mat frameBGR;
	if (!cap.read(frameBGR) || frameBGR.empty()) {
		std::cerr << "Could not read from camera" << std::endl;
		return EXIT_FAILURE;
	}
	int camW = frameBGR.cols;
	int camH = frameBGR.rows;

	std::cout << "Camera resolution: " << camW << " x " << camH << std::endl;

	// --- Create GL texture ---
	GLuint tex = 0;
	glGenTextures(1, &tex);
	glBindTexture(GL_TEXTURE_2D, tex);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB8, camW, camH, 0, GL_BGR, GL_UNSIGNED_BYTE, frameBGR.data);

	// --- Fullscreen quad ---
	const float quad[] = {
		-1.f, -1.f, 0.f, 0.f,
		 1.f, -1.f, 1.f, 0.f,
		 1.f,  1.f, 1.f, 1.f,
		-1.f,  1.f, 0.f, 1.f
	};
	const unsigned int idx[] = { 0,1,2, 2,3,0 };

	GLuint vao = 0, vbo = 0, ebo = 0;
	glGenVertexArrays(1, &vao);
	glBindVertexArray(vao);
	glGenBuffers(1, &vbo);
	glBindBuffer(GL_ARRAY_BUFFER, vbo);
	glBufferData(GL_ARRAY_BUFFER, sizeof(quad), quad, GL_STATIC_DRAW);
	glGenBuffers(1, &ebo);
	glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo);
	glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(idx), idx, GL_STATIC_DRAW);

	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

	// --- Program + uniforms ---
	GLint uTransformLoc = -1, uTexLoc = -1, uScaleLoc = -1, uFilterTypeLoc = -1, uPixelBlockLoc = -1, uRedThreshLoc = -1;
	GLuint prog = makeProgram(uTransformLoc, uTexLoc, uScaleLoc, uFilterTypeLoc, uPixelBlockLoc, uRedThreshLoc);
	if (!prog) {
		std::cerr << "Shader program creation failed" << std::endl;
		return EXIT_FAILURE;
	}

	// Filter state
	enum Filter { F_NONE = 0, F_PIXELATE = 1, F_SINCITY = 2, F_COMIC = 3, };
	bool useGPU = true; // toggle CPU/GPU
	int filter = F_NONE;
	int pixelBlock = 12; // default block size
	float redThresh = 0.6f;

	cv::Mat cpuOutBGR; // buffer for CPU-processed frames

	// Interactive state + callbacks
	AppState state;
	state.camW = camW; state.camH = camH;
	glfwSetWindowUserPointer(win, &state);
	glfwSetCursorPosCallback(win, cursorPosCB);
	glfwSetMouseButtonCallback(win, mouseButtonCB);
	glfwSetScrollCallback(win, scrollCB);

	// FPS meter
	auto t0 = std::chrono::steady_clock::now();
	int frames = 0;

	// --- Main loop ---
	while (!glfwWindowShouldClose(win)) {
		// Poll input
		glfwPollEvents();
		if (glfwGetKey(win, GLFW_KEY_ESCAPE) == GLFW_PRESS) glfwSetWindowShouldClose(win, GLFW_TRUE);

		if (glfwGetKey(win, GLFW_KEY_0) == GLFW_PRESS) filter = F_NONE;
		if (glfwGetKey(win, GLFW_KEY_1) == GLFW_PRESS) filter = F_PIXELATE;
		if (glfwGetKey(win, GLFW_KEY_2) == GLFW_PRESS) filter = F_SINCITY;
		if (glfwGetKey(win, GLFW_KEY_3) == GLFW_PRESS) filter = F_COMIC;
		if (glfwGetKey(win, GLFW_KEY_G) == GLFW_PRESS) useGPU = true;
		if (glfwGetKey(win, GLFW_KEY_C) == GLFW_PRESS) useGPU = false;
		if (glfwGetKey(win, GLFW_KEY_UP) == GLFW_PRESS) pixelBlock = std::min(256, pixelBlock + 1);
		if (glfwGetKey(win, GLFW_KEY_DOWN) == GLFW_PRESS) pixelBlock = std::max(1, pixelBlock - 1);
		if (glfwGetKey(win, GLFW_KEY_R) == GLFW_PRESS) { state.rot = 0.0f; state.scale = 1.0f; state.tx = 0.0f; state.ty = 0.0f; }

		// Grab a frame (BGR8)
		if (!cap.read(frameBGR) || frameBGR.empty()) {
			continue;
		}

		// CPU path: warp first with same transform, then apply CPU filter
		if (!useGPU) {
			cv::Mat warped;
			cv::Mat Ainv;
			uvForward_to_cvInverse2x3(state, camW, camH, Ainv);
			cv::warpAffine(frameBGR, warped, Ainv, cv::Size(camW, camH),
				cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));

			switch (filter) {
			case F_PIXELATE: pixelateCPU(warped, cpuOutBGR, pixelBlock); break;
			case F_SINCITY:  sinCityCPU(warped, cpuOutBGR, 0); break;
			case F_COMIC:    comicCPU(warped, cpuOutBGR); break;
			default:         cpuOutBGR = warped; break;
			}
			glBindTexture(GL_TEXTURE_2D, tex);
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, camW, camH, GL_BGR, GL_UNSIGNED_BYTE, cpuOutBGR.data);
		}
		else {
			// GPU path: upload raw camera frame; shader applies transform + effect
			glBindTexture(GL_TEXTURE_2D, tex);
			glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, camW, camH, GL_BGR, GL_UNSIGNED_BYTE, frameBGR.data);
		}

		// Compute aspect-preserving scale for letterboxing
		int winW, winH; glfwGetFramebufferSize(win, &winW, &winH);
		float winAspect = (float)winW / std::max(1, winH);
		float camAspect = (float)camW / (float)camH;
		float sx = 1.f, sy = 1.f;
		if (winAspect > camAspect) { sx = camAspect / winAspect; }
		else { sy = winAspect / camAspect; }

		// Stash for mouse mapping
		state.winW = winW; state.winH = winH;
		state.sx = sx; state.sy = sy;

		// Render
		glViewport(0, 0, winW, winH);
		glClearColor(0.05f, 0.05f, 0.08f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT);

		glUseProgram(prog);
		glUniform1i(uTexLoc, 0);
		glUniform2f(uScaleLoc, sx, sy);
		glUniform1i(uFilterTypeLoc, useGPU ? filter : 0); // 0 in CPU mode
		glUniform1f(uPixelBlockLoc, (float)pixelBlock);
		glUniform1f(uRedThreshLoc, redThresh);

		// Upload transform matrix (identity in CPU mode to avoid double-transform)
		if (!useGPU) {
			GLfloat I[9] = { 1,0,0, 0,1,0, 0,0,1 };
			glUniformMatrix3fv(uTransformLoc, 1, GL_FALSE, I);
		}
		else {
			Mat3 Muv = buildUvForward(state);
			GLfloat matData[9] = {
				Muv.a11, Muv.a21, Muv.a31,
				Muv.a12, Muv.a22, Muv.a32,
				Muv.a13, Muv.a23, Muv.a33
			};
			glUniformMatrix3fv(uTransformLoc, 1, GL_FALSE, matData);
		}

		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, tex);

		glBindVertexArray(vao);
		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

		glfwSwapBuffers(win);

		// FPS meter
		frames++;
		auto t1 = std::chrono::steady_clock::now();
		if (std::chrono::duration_cast<std::chrono::seconds>(t1 - t0).count() >= 1) {
			std::cout << "FPS: " << frames << (useGPU ? " [GPU]" : " [CPU]")
				<< " | Filter: " << (filter == 0 ? "None" : (filter == 1 ? "Pixelate" : (filter == 2 ? "SinCity" : "Comic")))
				<< " | Block: " << pixelBlock
				<< " | rot(deg): " << (state.rot * 180.0 / 3.14159265)
				<< " | scale: " << state.scale
				<< " | tx,ty: " << state.tx << "," << state.ty
				<< std::endl;
			frames = 0; t0 = t1;
		}
	}

	// Cleanup
	glDeleteProgram(prog);
	glDeleteBuffers(1, &ebo);
	glDeleteBuffers(1, &vbo);
	glDeleteVertexArrays(1, &vao);
	glDeleteTextures(1, &tex);

	glfwDestroyWindow(win);
	glfwTerminate();
	return EXIT_SUCCESS;
}
