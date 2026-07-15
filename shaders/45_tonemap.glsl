#version 330 core
// 45_tonemap.glsl — exposure + tonemapping curve.
//
// Slider annotations for the RoLux GUI:  // [min, max, step] description

uniform sampler2D uMain;

in  vec2 vUV;
out vec4 fragColor;

#define MODE      1      // [0, 2, 1] 0: Reinhard  1: ACES  2: Filmic
#define EXPOSURE  1.0    // [0, 4, 0.05] pre-tonemap gain

vec3 reinhard(vec3 c) { return c / (1.0 + c); }

vec3 aces(vec3 c) {
    // Narkowicz ACES fit
    const float a = 2.51, b = 0.03, c2 = 2.43, d = 0.59, e = 0.14;
    return clamp((c * (a * c + b)) / (c * (c2 * c + d) + e), 0.0, 1.0);
}

vec3 filmic(vec3 c) {
    // Uncharted2-ish
    vec3 x = max(vec3(0.0), c - 0.004);
    return (x * (6.2 * x + 0.5)) / (x * (6.2 * x + 1.7) + 0.06);
}

void main() {
    vec3 c = texture(uMain, vUV).rgb * EXPOSURE;
#if MODE == 0
    c = reinhard(c);
#elif MODE == 2
    c = filmic(c);
#else
    c = aces(c);
#endif
    fragColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}
