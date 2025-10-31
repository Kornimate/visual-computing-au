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