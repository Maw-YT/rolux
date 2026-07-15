#version 330 core
// 50_gamma.glsl — exposure / contrast / gamma tone adjust (chain end).
// Chains off uMain.
//
// Slider annotations for the RoLux GUI:  // [min, max, step] description

uniform sampler2D uMain;

in  vec2 vUV;
out vec4 fragColor;

#define EXPOSURE  1.0   // [0, 3, 0.02] linear brightness multiply
#define CONTRAST  1.0   // [0, 2, 0.02] pivot around mid-gray
#define GAMMA     1.00   // [0.3, 3, 0.02] >1 brightens mids, <1 darkens
#define PIVOT     0.5   // [0, 1, 0.01] contrast pivot point

void main() {
    vec3 c = texture(uMain, vUV).rgb;

    c *= EXPOSURE;                                 // exposure
    c = (c - PIVOT) * CONTRAST + PIVOT;            // contrast about pivot
    c = clamp(c, 0.0, 1.0);
    c = pow(c, vec3(1.0 / GAMMA));                 // gamma

    fragColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}
