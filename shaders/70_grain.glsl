#version 330 core
// 70_grain.glsl — animated film grain.
//
// Slider annotations for the RoLux GUI:  // [min, max, step] description

uniform sampler2D uMain;
uniform vec2 uResolution;
uniform float uTime;

in  vec2 vUV;
out vec4 fragColor;

#define AMOUNT   0.06   // [0, 0.3, 0.005] grain strength
#define SIZE     1.0    // [0.5, 4, 0.1] grain size (px)
#define LUMA     0.6    // [0, 1, 0.05] more grain in shadows

float hash(vec2 p) {
    p = fract(p * vec2(123.34, 456.21));
    p += dot(p, p + 45.32);
    return fract(p.x * p.y);
}

void main() {
    vec3 c = texture(uMain, vUV).rgb;
    vec2 gp = floor(vUV * uResolution / max(SIZE, 0.5));
    float n = hash(gp + fract(uTime) * 91.7) - 0.5;
    float l = dot(c, vec3(0.299, 0.587, 0.114));
    float shadow = mix(1.0, 1.0 - l, LUMA);
    fragColor = vec4(clamp(c + n * AMOUNT * shadow, 0.0, 1.0), 1.0);
}
