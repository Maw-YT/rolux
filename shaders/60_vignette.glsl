#version 330 core
// 60_vignette.glsl — darkened frame edges.
//
// Slider annotations for the RoLux GUI:  // [min, max, step] description

uniform sampler2D uMain;
uniform vec2 uResolution;

in  vec2 vUV;
out vec4 fragColor;

#define AMOUNT     0.5    // [0, 1, 0.02] darkening strength
#define RADIUS     0.75   // [0.2, 1.5, 0.02] clear-area radius
#define SOFTNESS   0.45   // [0.05, 1, 0.02] falloff width
#define ROUNDNESS  1.0    // [0, 1, 0.05] 1 = circular, 0 = follows aspect

void main() {
    vec3 c = texture(uMain, vUV).rgb;
    vec2 d = vUV - 0.5;
    float aspect = uResolution.x / max(uResolution.y, 1.0);
    d.x *= mix(aspect, 1.0, ROUNDNESS);   // ROUNDNESS=1 -> circular
    float dist = length(d) * 1.41421356;
    float v = smoothstep(RADIUS, RADIUS - SOFTNESS, dist);
    v = mix(1.0, v, AMOUNT);
    fragColor = vec4(c * v, 1.0);
}
