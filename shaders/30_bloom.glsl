#version 330 core
// 30_bloom.glsl — single-pass bright-pass bloom (golden-spiral gather).
// Chains off uMain. Extracts pixels above THRESHOLD, blurs them over a disk,
// and adds the glow back additively.
//
// Slider annotations for the RoLux GUI:  // [min, max, step] description

uniform sampler2D uMain;
uniform vec2 uResolution;

in  vec2 vUV;
out vec4 fragColor;

#define THRESHOLD  0.62   // [0, 1, 0.02] brightness cutoff for the glow
#define INTENSITY  0.65   // [0, 3, 0.05] added bloom strength
#define RADIUS     26.0   // [2, 96, 1] blur radius in pixels
#define SAMPLES    24      // [4, 48, 1] gather taps (quality vs cost)
#define TINT       1.0    // [0, 2, 0.05] keep source color (0 = white glow)

const float GOLDEN = 2.399963230; // radians

float luma(vec3 c) { return dot(c, vec3(0.2126, 0.7152, 0.0722)); }

void main() {
    vec3 base = texture(uMain, vUV).rgb;

    vec3  sum  = vec3(0.0);
    float wsum = 0.0;
    for (int i = 0; i < SAMPLES; i++) {
        float fi = float(i) + 0.5;
        float r  = sqrt(fi / float(SAMPLES));          // even disk coverage
        float a  = fi * GOLDEN;
        vec2  off = vec2(cos(a), sin(a)) * r * RADIUS / uResolution;

        vec3  c = texture(uMain, vUV + off).rgb;
        float l = luma(c);
        float b = max(l - THRESHOLD, 0.0);             // bright-pass amount
        vec3  bright = mix(vec3(b), c * (b / max(l, 1e-4)), TINT);

        float w = 1.0 - r;                             // center-weighted
        sum  += bright * w;
        wsum += w;
    }
    vec3 bloom = sum / max(wsum, 1e-4);
    fragColor = vec4(clamp(base + bloom * INTENSITY, 0.0, 1.0), 1.0);
}
