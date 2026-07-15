#version 330 core
// 90_letterbox.glsl — cinematic bars (final pass).
//
// Slider annotations for the RoLux GUI:  // [min, max, step] description

uniform sampler2D uMain;
uniform vec2 uResolution;

in  vec2 vUV;
out vec4 fragColor;

#define ASPECT     2.35   // [1.5, 2.8, 0.01] target aspect ratio
#define DARKNESS   0.0    // [0, 1, 0.02] bar brightness (0 = black)
#define FEATHER    0.004  // [0, 0.05, 0.001] soft bar edge

void main() {
    vec3 c = texture(uMain, vUV).rgb;
    float screenAspect = uResolution.x / max(uResolution.y, 1.0);
    // visible height fraction when fitting ASPECT into the screen width
    float visible = clamp(screenAspect / ASPECT, 0.0, 1.0);
    float bar = (1.0 - visible) * 0.5;              // top/bottom bar height in uv
    float m = smoothstep(bar - FEATHER, bar + FEATHER, vUV.y)
            * smoothstep(bar - FEATHER, bar + FEATHER, 1.0 - vUV.y);
    fragColor = vec4(mix(vec3(DARKNESS), c, m), 1.0);
}
