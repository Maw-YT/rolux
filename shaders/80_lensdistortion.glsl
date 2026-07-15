#version 330 core
// 80_lensdistortion.glsl — barrel / pincushion lens distortion.
//
// Slider annotations for the RoLux GUI:  // [min, max, step] description

uniform sampler2D uMain;

in  vec2 vUV;
out vec4 fragColor;

#define K1     0.10   // [-0.6, 0.6, 0.01] primary distortion (+ barrel, - pincushion)
#define K2     0.0    // [-0.6, 0.6, 0.01] secondary distortion
#define ZOOM   1.0    // [0.7, 1.4, 0.01] recenter / crop after distort

void main() {
    vec2 d = (vUV - 0.5) / ZOOM;
    float r2 = dot(d, d);
    vec2 uv = 0.5 + d * (1.0 + K1 * r2 + K2 * r2 * r2);
    if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) {
        fragColor = vec4(0.0, 0.0, 0.0, 1.0);
        return;
    }
    fragColor = vec4(texture(uMain, uv).rgb, 1.0);
}
