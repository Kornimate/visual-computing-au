#pragma once
#include <glad/glad.h>

class Shader {
public:
	static bool checkShader(GLuint obj, bool isProgram, const char* label);
	static GLuint Shader::makeProgram(GLint& uTransformLocOut, GLint& uTexLocOut, GLint& uScaleLocOut, GLint& uFilterTypeLocOut, GLint& uPixelBlockLocOut, GLint& uRedThreshLocOut);
};