#pragma once
#include <glad/glad.h>

class ShaderService {
public:
	static GLuint compileShader(GLenum type, const char* src);
	static GLuint createProgram(const char* vs, const char* fs);
};