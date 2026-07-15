#version 330 core
// 47_colorbalance.glsl — white balance (temperature / tint) + lift/gain.
//
// Slider annotations for the RoLux GUI:  // [min, max, step] description

uniform sampler2D uMain;

in  vec2 vUV;
out vec4 fragColor;

#define TEMPERATURE  0.0   // [-1, 1, 0.02] cool (-) .. warm (+)
#define TINT         0.0   // [-1, 1, 0.02] green (-) .. magenta (+)
#define LIFT         0.0   // [-0.5, 0.5, 0.01] shadow offset
#define GAIN         1.0   // [0, 2, 0.02] highlight multiply
#define GAMMA        1.0   // [0.3, 3, 0.02] midtone

void main() {
    vec3 c = texture(uMain, vUV).rgb;

    // temperature/tint as a simple channel bias
    c.r += TEMPERATURE * 0.10;
    c.b -= TEMPERATURE * 0.10;
    c.g += TINT * 0.10;
    c.r -= TINT * 0.05;
    c.b -= TINT * 0.05;

    // lift (shadows) / gain (highlights) / gamma (mids)
    c = c * GAIN + LIFT * (1.0 - c);
    c = pow(clamp(c, 0.0, 1.0), vec3(1.0 / GAMMA));

    fragColor = vec4(clamp(c, 0.0, 1.0), 1.0);
}
