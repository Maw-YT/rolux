#version 330 core
// 05_fxaa.glsl — fast approximate anti-aliasing (Lottes FXAA, luma edge).
// Runs early so later effects work on cleaned edges.
//
// Slider annotations for the RoLux GUI:  // [min, max, step] description

uniform sampler2D uMain;
uniform vec2 uResolution;

in  vec2 vUV;
out vec4 fragColor;

#define EDGE_MIN   0.0312   // [0, 0.1, 0.001] absolute min edge to touch
#define EDGE_MAX   0.125    // [0.03, 0.33, 0.005] relative edge threshold
#define SPAN_MAX   8.0      // [1, 16, 0.5] max blur search span (px)

float luma(vec3 c) { return dot(c, vec3(0.299, 0.587, 0.114)); }

void main() {
    vec2 px = 1.0 / uResolution;
    vec3 rgbM = texture(uMain, vUV).rgb;
    float lM  = luma(rgbM);
    float lNW = luma(texture(uMain, vUV + vec2(-1.0, -1.0) * px).rgb);
    float lNE = luma(texture(uMain, vUV + vec2( 1.0, -1.0) * px).rgb);
    float lSW = luma(texture(uMain, vUV + vec2(-1.0,  1.0) * px).rgb);
    float lSE = luma(texture(uMain, vUV + vec2( 1.0,  1.0) * px).rgb);

    float lMin = min(lM, min(min(lNW, lNE), min(lSW, lSE)));
    float lMax = max(lM, max(max(lNW, lNE), max(lSW, lSE)));
    if (lMax - lMin < max(EDGE_MIN, lMax * EDGE_MAX)) {
        fragColor = vec4(rgbM, 1.0);
        return;
    }

    vec2 dir = vec2(
        -((lNW + lNE) - (lSW + lSE)),
         ((lNW + lSW) - (lNE + lSE))
    );
    float reduce = max((lNW + lNE + lSW + lSE) * (0.25 * (1.0 / 8.0)), 1.0 / 128.0);
    float rcpMin = 1.0 / (min(abs(dir.x), abs(dir.y)) + reduce);
    dir = clamp(dir * rcpMin, -SPAN_MAX, SPAN_MAX) * px;

    vec3 rgbA = 0.5 * (
        texture(uMain, vUV + dir * (1.0 / 3.0 - 0.5)).rgb +
        texture(uMain, vUV + dir * (2.0 / 3.0 - 0.5)).rgb
    );
    vec3 rgbB = rgbA * 0.5 + 0.25 * (
        texture(uMain, vUV + dir * -0.5).rgb +
        texture(uMain, vUV + dir *  0.5).rgb
    );
    float lB = luma(rgbB);
    fragColor = vec4((lB < lMin || lB > lMax) ? rgbA : rgbB, 1.0);
}
