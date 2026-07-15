#version 330 core
// 65_chromatic.glsl — chromatic aberration (RGB fringe toward edges).
//
// Slider annotations for the RoLux GUI:  // [min, max, step] description

uniform sampler2D uMain;
uniform vec2 uResolution;

in  vec2 vUV;
out vec4 fragColor;

#define AMOUNT   2.0    // [0, 10, 0.1] max channel shift (px)
#define RADIAL   1      // [0, 1, 1] 1: scale by distance from center

void main() {
    vec2 dir = vUV - 0.5;
#if RADIAL == 1
    float scale = length(dir) * 2.0;
#else
    float scale = 1.0;
#endif
    vec2 shift = normalize(dir + 1e-6) * (AMOUNT * scale) / uResolution;

    float r = texture(uMain, vUV + shift).r;
    float g = texture(uMain, vUV).g;
    float b = texture(uMain, vUV - shift).b;
    fragColor = vec4(r, g, b, 1.0);
}
