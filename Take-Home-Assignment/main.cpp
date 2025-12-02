#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>

// --------------------------------------------------------
// Constants
// --------------------------------------------------------
const int DRAW_W = 800;
const int DRAW_H = 600;

// --------------------------------------------------------
// Data structures
// --------------------------------------------------------
struct DetectedShape {
    std::string label;              // "circle", "square", "rectangle", "triangle", "pentagon", "polygon", "none"
    std::vector<cv::Point> polygon; // approximated contour
};

struct Mesh {
    std::vector<float> vertices;        // x, y, z
    std::vector<unsigned int> indices;  // triangle indices
};

struct GLMesh {
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint ebo = 0;
    GLsizei indexCount = 0;
};

struct SceneObject {
    GLMesh* mesh;   // pointer to mesh (predefined or dynamic)
    float tx, ty, tz; // translation
};

struct AppState {
    bool drawing = false;
    double lastX = 0.0;
    double lastY = 0.0;
    bool needUpdate = true;
    cv::Mat canvas;

    GLuint tex = 0;
    GLuint quadVAO = 0;
    GLuint quadVBO = 0;
    GLuint quadEBO = 0;
};

// --------------------------------------------------------
// Globals
// --------------------------------------------------------
AppState g_state;

GLMesh gCubeMesh;
GLMesh gCuboidMesh;
GLMesh gSphereMesh;
GLMesh gPyramidMesh;
GLMesh gPentagonPrismMesh;
std::vector<GLMesh> gDynamicMeshes;   // meshes created from extrusion (X)
std::vector<SceneObject> gObjects;    // all objects in 3D scene

GLuint gProg2D = 0;
GLuint gProg3D = 0;

// --------------------------------------------------------
// Shader helpers
// --------------------------------------------------------
GLuint compileShader(GLenum type, const char* src) {
    GLuint sh = glCreateShader(type);
    glShaderSource(sh, 1, &src, nullptr);
    glCompileShader(sh);
    GLint ok;
    glGetShaderiv(sh, GL_COMPILE_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetShaderInfoLog(sh, 1024, nullptr, log);
        std::cerr << "Shader compile error:\n" << log << "\n";
    }
    return sh;
}

GLuint createProgram(const char* vs, const char* fs) {
    GLuint v = compileShader(GL_VERTEX_SHADER, vs);
    GLuint f = compileShader(GL_FRAGMENT_SHADER, fs);
    GLuint p = glCreateProgram();
    glAttachShader(p, v);
    glAttachShader(p, f);
    glLinkProgram(p);
    GLint ok;
    glGetProgramiv(p, GL_LINK_STATUS, &ok);
    if (!ok) {
        char log[1024];
        glGetProgramInfoLog(p, 1024, nullptr, log);
        std::cerr << "Program link error:\n" << log << "\n";
    }
    glDeleteShader(v);
    glDeleteShader(f);
    return p;
}

// --------------------------------------------------------
// Shape detection (square vs rectangle, circle vs polygon)
// --------------------------------------------------------
DetectedShape detectShapeWithPolygon(const cv::Mat& img) {
    cv::Mat gray, blurImg, thresh;
    cv::cvtColor(img, gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(gray, blurImg, cv::Size(5, 5), 0);
    cv::threshold(blurImg, thresh, 200, 255, cv::THRESH_BINARY_INV);

    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    if (contours.empty()) {
        return { "none", {} };
    }

    // Largest contour
    double maxArea = 0.0;
    int maxIdx = -1;
    for (int i = 0; i < (int)contours.size(); ++i) {
        double a = cv::contourArea(contours[i]);
        if (a > maxArea) {
            maxArea = a;
            maxIdx = i;
        }
    }

    auto& c = contours[maxIdx];

    std::vector<cv::Point> approx;
    cv::approxPolyDP(c, approx, 0.02 * cv::arcLength(c, true), true);

    int n = (int)approx.size();
    std::string label = "polygon";

    if (n == 3) {
        label = "triangle";
    }
    else if (n == 4) {
        // Distinguish square vs rectangle using bounding box aspect ratio
        cv::Rect r = cv::boundingRect(c);
        float ar = (float)r.width / (float)r.height;
        if (ar < 1.0f) ar = 1.0f / ar; // make >= 1
        if (ar < 1.1f) label = "square";      // roughly equal sides
        else           label = "rectangle";   // more rectangular
    }
    else if (n == 5) {
        label = "pentagon";
    }
    else {
        // check for circle-like
        double peri = cv::arcLength(c, true);
        double area = cv::contourArea(c);
        double circularity = 4 * CV_PI * (area / (peri * peri));
        if (circularity > 0.75) label = "circle";
        else label = "polygon";
    }

    return { label, approx };
}

// --------------------------------------------------------
// Extrusion (X key) - Y-axis prism from arbitrary polygon
// --------------------------------------------------------
Mesh extrudeY(const std::vector<cv::Point>& poly, float height) {
    Mesh m;
    if (poly.size() < 3) return m;

    int n = (int)poly.size();
    float half = height * 0.5f;

    // Center in pixel coordinates
    float cx = 0.0f, cy = 0.0f;
    for (auto& p : poly) {
        cx += (float)p.x;
        cy += (float)p.y;
    }
    cx /= n;
    cy /= n;

    float scale = 0.0015f;  // pixel -> world scaling

    // Bottom ring (y = -half)
    for (auto& p : poly) {
        float x = (p.x - cx) * scale;
        float z = (cy - p.y) * scale; // flip Y
        m.vertices.push_back(x);
        m.vertices.push_back(-half);
        m.vertices.push_back(z);
    }

    // Top ring (y = +half)
    for (auto& p : poly) {
        float x = (p.x - cx) * scale;
        float z = (cy - p.y) * scale;
        m.vertices.push_back(x);
        m.vertices.push_back(+half);
        m.vertices.push_back(z);
    }

    // Bottom face (triangle fan)
    for (int i = 1; i < n - 1; ++i) {
        m.indices.push_back(0);
        m.indices.push_back(i + 1);
        m.indices.push_back(i);
    }

    // Top face (triangle fan)
    int off = n;
    for (int i = 1; i < n - 1; ++i) {
        m.indices.push_back(off);
        m.indices.push_back(off + i);
        m.indices.push_back(off + i + 1);
    }

    // Side walls
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;

        int bi = i;
        int bj = j;
        int ti = i + n;
        int tj = j + n;

        // First triangle
        m.indices.push_back(bi);
        m.indices.push_back(ti);
        m.indices.push_back(bj);

        // Second triangle
        m.indices.push_back(bj);
        m.indices.push_back(ti);
        m.indices.push_back(tj);
    }

    return m;
}

// --------------------------------------------------------
// Primitive mesh builders (predefined shapes)
// --------------------------------------------------------
Mesh createBoxMesh(float sx, float sy, float sz) {
    Mesh m;
    float x = sx * 0.5f;
    float y = sy * 0.5f;
    float z = sz * 0.5f;

    // 8 vertices
    float v[] = {
        -x, -y, -z,
         x, -y, -z,
         x,  y, -z,
        -x,  y, -z,
        -x, -y,  z,
         x, -y,  z,
         x,  y,  z,
        -x,  y,  z
    };
    m.vertices.assign(v, v + 8 * 3);

    unsigned int idx[] = {
        // front (z+)
        4,5,6,  6,7,4,
        // back (z-)
        1,0,3,  3,2,1,
        // left (x-)
        0,4,7,  7,3,0,
        // right (x+)
        5,1,2,  2,6,5,
        // top (y+)
        3,7,6,  6,2,3,
        // bottom (y-)
        0,1,5,  5,4,0
    };
    m.indices.assign(idx, idx + 36);
    return m;
}

Mesh createPyramidMesh(float baseSize, float height) {
    Mesh m;
    float h = height;
    float b = baseSize * 0.5f;

    // Base triangle on y = -h/2
    // Equilateral-ish triangle in XZ
    float yBase = -h * 0.5f;
    float yTop = +h * 0.5f;

    // 0,1,2: base; 3: apex
    std::vector<float> v = {
        -b, yBase, -b,
         b, yBase, -b,
         0, yBase,  b,
         0, yTop,   0
    };
    m.vertices = v;

    // Base (triangle)
    m.indices.push_back(0);
    m.indices.push_back(1);
    m.indices.push_back(2);

    // Sides
    m.indices.push_back(0);
    m.indices.push_back(1);
    m.indices.push_back(3);

    m.indices.push_back(1);
    m.indices.push_back(2);
    m.indices.push_back(3);

    m.indices.push_back(2);
    m.indices.push_back(0);
    m.indices.push_back(3);

    return m;
}

Mesh createSphereMesh(float radius, int slices, int stacks) {
    Mesh m;
    for (int i = 0; i <= stacks; ++i) {
        float v = (float)i / (float)stacks;
        float phi = v * CV_PI; // 0..PI
        float y = std::cos(phi);
        float r = std::sin(phi);
        for (int j = 0; j <= slices; ++j) {
            float u = (float)j / (float)slices;
            float theta = u * 2.0f * CV_PI; // 0..2PI
            float x = r * std::cos(theta);
            float z = r * std::sin(theta);
            m.vertices.push_back(radius * x);
            m.vertices.push_back(radius * y);
            m.vertices.push_back(radius * z);
        }
    }

    for (int i = 0; i < stacks; ++i) {
        for (int j = 0; j < slices; ++j) {
            int i0 = i * (slices + 1) + j;
            int i1 = i0 + 1;
            int i2 = i0 + (slices + 1);
            int i3 = i2 + 1;
            m.indices.push_back(i0);
            m.indices.push_back(i2);
            m.indices.push_back(i1);
            m.indices.push_back(i1);
            m.indices.push_back(i2);
            m.indices.push_back(i3);
        }
    }
    return m;
}

Mesh createPentagonPrismMesh(float radius, float height) {
    Mesh m;
    int n = 5;
    float h = height * 0.5f;

    // Bottom ring
    for (int i = 0; i < n; ++i) {
        float a = (2.0f * CV_PI * i) / n;
        float x = radius * std::cos(a);
        float z = radius * std::sin(a);
        m.vertices.push_back(x);
        m.vertices.push_back(-h);
        m.vertices.push_back(z);
    }

    // Top ring
    for (int i = 0; i < n; ++i) {
        float a = (2.0f * CV_PI * i) / n;
        float x = radius * std::cos(a);
        float z = radius * std::sin(a);
        m.vertices.push_back(x);
        m.vertices.push_back(+h);
        m.vertices.push_back(z);
    }

    // Bottom face
    for (int i = 1; i < n - 1; ++i) {
        m.indices.push_back(0);
        m.indices.push_back(i + 1);
        m.indices.push_back(i);
    }

    // Top face
    int off = n;
    for (int i = 1; i < n - 1; ++i) {
        m.indices.push_back(off);
        m.indices.push_back(off + i);
        m.indices.push_back(off + i + 1);
    }

    // Walls
    for (int i = 0; i < n; ++i) {
        int j = (i + 1) % n;
        int bi = i;
        int bj = j;
        int ti = i + n;
        int tj = j + n;

        m.indices.push_back(bi);
        m.indices.push_back(ti);
        m.indices.push_back(bj);
        m.indices.push_back(bj);
        m.indices.push_back(ti);
        m.indices.push_back(tj);
    }

    return m;
}

// --------------------------------------------------------
// Upload Mesh to GPU
// --------------------------------------------------------
GLMesh uploadMesh(const Mesh& m) {
    GLMesh gm;
    if (m.vertices.empty() || m.indices.empty()) return gm;

    gm.indexCount = (GLsizei)m.indices.size();

    glGenVertexArrays(1, &gm.vao);
    glGenBuffers(1, &gm.vbo);
    glGenBuffers(1, &gm.ebo);

    glBindVertexArray(gm.vao);

    glBindBuffer(GL_ARRAY_BUFFER, gm.vbo);
    glBufferData(GL_ARRAY_BUFFER, m.vertices.size() * sizeof(float),
        m.vertices.data(), GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, gm.ebo);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, m.indices.size() * sizeof(unsigned int),
        m.indices.data(), GL_STATIC_DRAW);

    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 3 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);

    glBindVertexArray(0);
    return gm;
}

// --------------------------------------------------------
// Callbacks for 2D window
// --------------------------------------------------------
void mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
    (void)window; (void)mods;
    if (button == GLFW_MOUSE_BUTTON_LEFT) {
        if (action == GLFW_PRESS) {
            g_state.drawing = true;
            glfwGetCursorPos(window, &g_state.lastX, &g_state.lastY);
        }
        else if (action == GLFW_RELEASE) {
            g_state.drawing = false;
        }
    }
}

void cursor_position_callback(GLFWwindow* window, double xpos, double ypos) {
    (void)window;
    if (!g_state.drawing) return;

    double x = std::max(0.0, std::min(xpos, (double)DRAW_W - 1));
    double y = std::max(0.0, std::min(ypos, (double)DRAW_H - 1));

    cv::Point p1((int)g_state.lastX, (int)g_state.lastY);
    cv::Point p2((int)x, (int)y);
    cv::line(g_state.canvas, p1, p2, cv::Scalar(0, 0, 0), 3, cv::LINE_AA);

    g_state.lastX = x;
    g_state.lastY = y;
    g_state.needUpdate = true;
}

// Build a simple MVP (here just rotation * translation, no perspective)
void buildRotY(float angle, float tx, float ty, float tz, float out[16]) {
    float c = std::cos(angle);
    float s = std::sin(angle);

    // Column-major
    out[0] = c;   out[4] = 0.0f; out[8] = s;   out[12] = tx;
    out[1] = 0.0; out[5] = 1.0f; out[9] = 0.0; out[13] = ty;
    out[2] = -s;  out[6] = 0.0f; out[10] = c;   out[14] = tz;
    out[3] = 0.0; out[7] = 0.0f; out[11] = 0.0; out[15] = 1.0f;
}

void key_callback_2D(GLFWwindow* window, int key, int scancode, int action, int mods) {
    (void)window; (void)scancode; (void)mods;
    if (action != GLFW_PRESS) return;

    if (key == GLFW_KEY_C) {
        g_state.canvas.setTo(cv::Scalar(255, 255, 255));
        g_state.needUpdate = true;
        std::cout << "Canvas cleared.\n";
    }

    if (key == GLFW_KEY_R) {
        // Shape recognition -> predefined primitive
        DetectedShape ds = detectShapeWithPolygon(g_state.canvas);
        std::cout << "Recognized: " << ds.label << "\n";

        GLMesh* chosen = nullptr;

        if (ds.label == "circle")       chosen = &gSphereMesh;
        else if (ds.label == "square")  chosen = &gCubeMesh;
        else if (ds.label == "rectangle") chosen = &gCuboidMesh;
        else if (ds.label == "triangle")  chosen = &gPyramidMesh;
        else if (ds.label == "pentagon")  chosen = &gPentagonPrismMesh;
        else {
            std::cout << "No predefined 3D shape for label: " << ds.label << "\n";
        }

        if (chosen && chosen->vao != 0) {
            // Position new object in grid
            int idx = (int)gObjects.size();
            int row = idx / 5;
            int col = idx % 5;
            float tx = (col - 2) * 0.8f;
            float tz = -row * 0.8f;
            float ty = 0.0f;

            gObjects.push_back({ chosen, tx, ty, tz });
            std::cout << "Added 3D primitive to scene.\n";
        }
    }

    if (key == GLFW_KEY_X) {
        // Extrusion of actual user-drawn polygon
        DetectedShape ds = detectShapeWithPolygon(g_state.canvas);
        if (ds.polygon.empty()) {
            std::cout << "No shape to extrude.\n";
            return;
        }
        std::cout << "Extruding drawn polygon (label: " << ds.label << ")\n";

        Mesh m = extrudeY(ds.polygon, 0.6f);
        GLMesh gm = uploadMesh(m);
        if (gm.vao != 0) {
            gDynamicMeshes.push_back(gm);
            GLMesh* ptr = &gDynamicMeshes.back();

            int idx = (int)gObjects.size();
            int row = idx / 5;
            int col = idx % 5;
            float tx = (col - 2) * 0.8f;
            float tz = -row * 0.8f;
            float ty = 0.0f;

            gObjects.push_back({ ptr, tx, ty, tz });
            std::cout << "Added extruded shape to scene.\n";
        }
        else {
            std::cout << "Failed to upload extruded mesh.\n";
        }
    }
}

// --------------------------------------------------------
// Main
// --------------------------------------------------------
int main() {
    if (!glfwInit()) {
        std::cerr << "Failed to init GLFW.\n";
        return -1;
    }

    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);

    // ------------- 2D WINDOW -------------
    GLFWwindow* win2D = glfwCreateWindow(DRAW_W, DRAW_H, "2D Drawing", nullptr, nullptr);
    if (!win2D) {
        std::cerr << "Failed to create 2D window.\n";
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(win2D);
    if (!gladLoadGLLoader((GLADloadproc)glfwGetProcAddress)) {
        std::cerr << "Failed to init GLAD.\n";
        glfwTerminate();
        return -1;
    }

    g_state.canvas = cv::Mat(DRAW_H, DRAW_W, CV_8UC3, cv::Scalar(255, 255, 255));

    glfwSetMouseButtonCallback(win2D, mouse_button_callback);
    glfwSetCursorPosCallback(win2D, cursor_position_callback);
    glfwSetKeyCallback(win2D, key_callback_2D);

    // 2D textured quad shader
    const char* vs2D = R"(
        #version 330 core
        layout(location=0) in vec2 aPos;
        layout(location=1) in vec2 aTex;
        out vec2 TexCoord;
        void main() {
            TexCoord = aTex;
            gl_Position = vec4(aPos, 0.0, 1.0);
        }
    )";

    const char* fs2D = R"(
        #version 330 core
        in vec2 TexCoord;
        out vec4 FragColor;
        uniform sampler2D uTex;
        void main() {
            FragColor = texture(uTex, TexCoord);
        }
    )";

    gProg2D = createProgram(vs2D, fs2D);

    float quadVerts[] = {
        // pos      // tex
        -1.0f, -1.0f, 0.0f, 0.0f,
         1.0f, -1.0f, 1.0f, 0.0f,
         1.0f,  1.0f, 1.0f, 1.0f,
        -1.0f,  1.0f, 0.0f, 1.0f
    };
    unsigned int quadIdx[] = { 0,1,2, 2,3,0 };

    glGenVertexArrays(1, &g_state.quadVAO);
    glGenBuffers(1, &g_state.quadVBO);
    glGenBuffers(1, &g_state.quadEBO);

    glBindVertexArray(g_state.quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, g_state.quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVerts), quadVerts, GL_STATIC_DRAW);

    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, g_state.quadEBO);
    glBufferData(GL_ELEMENT_ARRAY_BUFFER, sizeof(quadIdx), quadIdx, GL_STATIC_DRAW);

    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));
    glEnableVertexAttribArray(1);

    // texture for canvas
    glGenTextures(1, &g_state.tex);
    glBindTexture(GL_TEXTURE_2D, g_state.tex);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, DRAW_W, DRAW_H, 0,
        GL_BGR, GL_UNSIGNED_BYTE, g_state.canvas.data);

    // ------------- 3D WINDOW (shared context) -------------
    GLFWwindow* win3D = glfwCreateWindow(600, 600, "3D Scene", nullptr, win2D);
    if (!win3D) {
        std::cerr << "Failed to create 3D window.\n";
        glfwDestroyWindow(win2D);
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(win3D);
    // GLAD already initialized for the shared context

    const char* vs3D = R"(
        #version 330 core
        layout(location=0) in vec3 aPos;
        uniform mat4 uMVP;
        void main() {
            gl_Position = uMVP * vec4(aPos, 1.0);
        }
    )";

    const char* fs3D = R"(
        #version 330 core
        out vec4 FragColor;
        void main() {
            FragColor = vec4(0.8, 0.85, 0.95, 1.0);
        }
    )";

    gProg3D = createProgram(vs3D, fs3D);

    // Build predefined meshes
    Mesh cubeMesh = createBoxMesh(0.6f, 0.6f, 0.6f);      // for squares
    Mesh cuboidMesh = createBoxMesh(1.0f, 0.6f, 0.6f);      // fixed cuboid for rectangles
    Mesh sphereMesh = createSphereMesh(0.4f, 24, 16);       // for circles
    Mesh pyramidMesh = createPyramidMesh(0.8f, 0.8f);        // for triangles
    Mesh pentPrismMesh = createPentagonPrismMesh(0.5f, 0.8f);  // for pentagons

    gCubeMesh = uploadMesh(cubeMesh);
    gCuboidMesh = uploadMesh(cuboidMesh);
    gSphereMesh = uploadMesh(sphereMesh);
    gPyramidMesh = uploadMesh(pyramidMesh);
    gPentagonPrismMesh = uploadMesh(pentPrismMesh);

    // Set initial context back to 2D window
    glfwMakeContextCurrent(win2D);

    // ------------- Main loop -------------
    while (!glfwWindowShouldClose(win2D) &&
        !glfwWindowShouldClose(win3D)) {

        // ----------- 2D window -----------
        glfwMakeContextCurrent(win2D);
        int fbw2, fbh2;
        glfwGetFramebufferSize(win2D, &fbw2, &fbh2);
        glViewport(0, 0, fbw2, fbh2);

        if (g_state.needUpdate) {
            // Create a display image with crosshair
            cv::Mat display = g_state.canvas.clone();
            cv::line(display, cv::Point(DRAW_W / 2, 0), cv::Point(DRAW_W / 2, DRAW_H), cv::Scalar(200, 200, 200), 1);
            cv::line(display, cv::Point(0, DRAW_H / 2), cv::Point(DRAW_W, DRAW_H / 2), cv::Scalar(200, 200, 200), 1);

            cv::Mat flipped;
            cv::flip(display, flipped, 0);

            glBindTexture(GL_TEXTURE_2D, g_state.tex);
            glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, DRAW_W, DRAW_H,
                GL_BGR, GL_UNSIGNED_BYTE, flipped.data);

            g_state.needUpdate = false;
        }

        glClearColor(0.2f, 0.25f, 0.3f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT);

        glUseProgram(gProg2D);
        glUniform1i(glGetUniformLocation(gProg2D, "uTex"), 0);
        glActiveTexture(GL_TEXTURE0);
        glBindTexture(GL_TEXTURE_2D, g_state.tex);

        glBindVertexArray(g_state.quadVAO);
        glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_INT, 0);

        glfwSwapBuffers(win2D);

        // ----------- 3D window -----------
        glfwMakeContextCurrent(win3D);
        int fbw3, fbh3;
        glfwGetFramebufferSize(win3D, &fbw3, &fbh3);
        glViewport(0, 0, fbw3, fbh3);

        glEnable(GL_DEPTH_TEST);
        glClearColor(0.1f, 0.12f, 0.15f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glUseProgram(gProg3D);

        float time = (float)glfwGetTime();
        for (size_t i = 0; i < gObjects.size(); ++i) {
            SceneObject& obj = gObjects[i];
            float M[16];
            // global rotation over time + object translation
            buildRotY(time * 0.5f, obj.tx, obj.ty, obj.tz, M);
            glUniformMatrix4fv(glGetUniformLocation(gProg3D, "uMVP"), 1, GL_FALSE, M);

            glBindVertexArray(obj.mesh->vao);
            glDrawElements(GL_TRIANGLES, obj.mesh->indexCount, GL_UNSIGNED_INT, 0);
        }

        glfwSwapBuffers(win3D);

        // Handle events for both windows
        glfwPollEvents();
    }

    glfwDestroyWindow(win3D);
    glfwDestroyWindow(win2D);
    glfwTerminate();
    return 0;
}
