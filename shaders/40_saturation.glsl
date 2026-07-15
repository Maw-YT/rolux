#version 330 core
// 40_saturation.glsl — saturation + vibrance color grade.
// Chains off uMain.
//
// Slider annotations for the RoLux GUI:  // [min, max, step] description

uniform sampler2D uMain;

in  vec2 vUV;
out vec4 fragColor;

#define SATURATION  1.20   // [0, 3, 0.02] 1 = neutral, 0 = grayscale
#define VIBRANCE    0.25   // [-1, 1, 0.02] boosts muted colors, spares vivid ones

float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

void main() {
    vec3 c = texture(uMain, vUV).rgb;

    // global saturation about luma
    vec3 g = vec3(luma(c));
    vec3 s = mix(g, c, SATURATION);

    // vibrance: scale the boost by how unsaturated the pixel already is
    float mx  = max(s.r, max(s.g, s.b));
    float mn  = min(s.r, min(s.g, s.b));
    float sat = mx - mn;                       // 0 = gray, 1 = pure
    float amt = 1.0 + VIBRANCE * (1.0 - sat);
    vec3  out3 = mix(vec3(luma(s)), s, amt);

    fragColor = vec4(clamp(out3, 0.0, 1.0), 1.0);
}
