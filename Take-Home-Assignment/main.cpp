#include <iostream>
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <opencv2/opencv.hpp>

// Build a simple MVP (rotation around Y + translation)
void buildRotY(float angle, float tx, float ty, float tz, float out[16]) {
    float c = std::cos(angle);
    float s = std::sin(angle);

    // Column-major
    out[0] = c;   out[4] = 0.0f; out[8] = s;   out[12] = tx;
    out[1] = 0.0; out[5] = 1.0f; out[9] = 0.0; out[13] = ty;
    out[2] = -s;  out[6] = 0.0f; out[10] = c;   out[14] = tz;
    out[3] = 0.0; out[7] = 0.0f; out[11] = 0.0; out[15] = 1.0f;
}

// --------------------------------------------------------
// Main
// --------------------------------------------------------
int main() {
    // ------------- 3D WINDOW (shared context) -------------
    GLFWwindow* win3D = glfwCreateWindow(600, 600, "3D Scene", nullptr, win2D);
    if (!win3D) {
        std::cerr << "Failed to create 3D window.\n";
        glfwDestroyWindow(win2D);
        glfwTerminate();
        return -1;
    }

    glfwMakeContextCurrent(win3D);
    // GLAD already initialized for this shared context

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

    // Back to 2D as main context
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
//#include "Views/window2D.h"
//#include "Views/window3D.h"
//#include "app.h"
//
//int main() {
//  srand((unsigned)time(nullptr));
//  cv::utils::logging::setLogLevel(cv::utils::logging::LOG_LEVEL_SILENT);
// 
//	App* app = new App(new Window2D(), new Window3D());
//
//	app->initialize();
//	app->run();
//
//
//	delete app;
//	return 0;
//}