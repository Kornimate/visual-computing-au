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