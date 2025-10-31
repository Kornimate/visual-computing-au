#include <glad/glad.h>
#include <vector>
#include <iostream>
#include "Shader.h"
#include "../Services/FileService.h"

// Check & log shader compilation errors 
bool Shader::checkShader(GLuint obj, bool isProgram, const char* label) {
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

// GLSL Program (that supports also GPU filters)
GLuint Shader::makeProgram(GLint& uTransformLocOut, GLint& uTexLocOut, GLint& uScaleLocOut,
    GLint& uFilterTypeLocOut, GLint& uPixelBlockLocOut, GLint& uRedThreshLocOut)
{
    std::string vsSrcString = FileService::ReadFileContent("./Resources/VertexShader.vert");
    std::string fsSrcString = FileService::ReadFileContent("./Resources/FragmentShader.frag");

    const char* vsSrc = vsSrcString.c_str();
    const char* fsSrc = fsSrcString.c_str();


    GLuint vs = glCreateShader(GL_VERTEX_SHADER);
    glShaderSource(vs, 1, &vsSrc, nullptr);
    glCompileShader(vs);
    if (!Shader::checkShader(vs, false, "Vertex Shader")) return 0;

    GLuint fs = glCreateShader(GL_FRAGMENT_SHADER);
    glShaderSource(fs, 1, &fsSrc, nullptr);
    glCompileShader(fs);
    if (!Shader::checkShader(fs, false, "Fragment Shader")) return 0;

    GLuint prog = glCreateProgram();
    glAttachShader(prog, vs);
    glAttachShader(prog, fs);
    glLinkProgram(prog);
    if (!Shader::checkShader(prog, true, "Program")) return 0;

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