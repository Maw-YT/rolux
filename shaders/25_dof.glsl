#version 330 core
// 25_dof.glsl — depth of field (circle-of-confusion disk blur).
// Blurs by distance from the focus plane using the depth buffer.
//
// Slider annotations for the RoLux GUI:  // [min, max, step] description

uniform sampler2D uMain;
uniform sampler2D uDepthF;
uniform vec2 uResolution;

in  vec2 vUV;
out vec4 fragColor;

#define DEPTH_NEAR_IS_ONE 1     // [0, 1, 1] DA-V2 disparity: 1 = near
#define FOCUS     0.35   // [0, 1, 0.01] focus distance (0 near .. 1 far)
#define RANGE     0.12   // [0.01, 1, 0.01] in-focus depth range
#define MAX_BLUR  6.0    // [0, 20, 0.5] max blur radius (px)
#define SAMPLES   16     // [4, 32, 1] bokeh taps

const float GOLDEN = 2.399963230;

float far(vec2 uv) {
    float d = texture(uDepthF, uv).r;
#if DEPTH_NEAR_IS_ONE
    d = 1.0 - d;
#endif
    return d;
}

void main() {
    vec3 base = texture(uMain, vUV).rgb;
    float coc = clamp((abs(far(vUV) - FOCUS) - RANGE) / max(RANGE, 1e-3), 0.0, 1.0);
    float radius = coc * MAX_BLUR;
    if (radius < 0.5) { fragColor = vec4(base, 1.0); return; }

    vec3 acc = base;
    float wsum = 1.0;
    for (int i = 0; i < SAMPLES; i++) {
        float fi = float(i) + 0.5;
        float r  = sqrt(fi / float(SAMPLES)) * radius;
        float a  = fi * GOLDEN;
        vec2 off = vec2(cos(a), sin(a)) * r / uResolution;
        // weight blurred (out-of-focus) samples more so sharp foreground
        // doesn't smear onto blurred background
        float sc = clamp((abs(far(vUV + off) - FOCUS) - RANGE) / max(RANGE, 1e-3), 0.0, 1.0);
        float w  = max(sc, 0.05);
        acc  += texture(uMain, vUV + off).rgb * w;
        wsum += w;
    }
    fragColor = vec4(acc / wsum, 1.0);
}
