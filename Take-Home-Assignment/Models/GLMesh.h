#pragma once
#include <glad/glad.h>
#include <glm/glm.hpp>

struct GLMesh {
    GLuint vao = 0;
    GLuint vbo = 0;
    GLuint ebo = 0;
    GLsizei indexCount = 0;
    glm::vec3 aabbMin{ 0.0f };
    glm::vec3 aabbMax{ 0.0f };
};