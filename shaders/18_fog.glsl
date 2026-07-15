#version 330 core
// 18_fog.glsl — depth-based distance fog.
//
// Slider annotations for the RoLux GUI:  // [min, max, step] description

uniform sampler2D uMain;
uniform sampler2D uDepthF;

in  vec2 vUV;
out vec4 fragColor;

#define DEPTH_NEAR_IS_ONE 1     // [0, 1, 1] DA-V2 disparity: 1 = near
#define DENSITY   1.4    // [0, 6, 0.05] fog thickness
#define START     0.05   // [0, 1, 0.01] distance where fog begins
#define FOG_R     0.62   // [0, 1, 0.01] fog color R
#define FOG_G     0.70   // [0, 1, 0.01] fog color G
#define FOG_B     0.82   // [0, 1, 0.01] fog color B

void main() {
    vec3 base = texture(uMain, vUV).rgb;
    float d = texture(uDepthF, vUV).r;
#if DEPTH_NEAR_IS_ONE
    d = 1.0 - d;                 // 0 = near, 1 = far
#endif
    float far = max(d - START, 0.0);
    float f = 1.0 - exp(-DENSITY * far);
    fragColor = vec4(mix(base, vec3(FOG_R, FOG_G, FOG_B), clamp(f, 0.0, 1.0)), 1.0);
}
