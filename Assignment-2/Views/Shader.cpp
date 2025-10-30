#include <glad/glad.h>
#include <vector>
#include <iostream>
#include "Shader.h"

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