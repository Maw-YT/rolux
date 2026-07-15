#version 330 core
// 20_sharpen.glsl — unsharp-mask sharpening.
// Chains off uMain (previous pass), so it sharpens whatever came before.
//
// Slider annotations for the RoLux GUI:  // [min, max, step] description

uniform sampler2D uMain;
uniform vec2 uResolution;

in  vec2 vUV;
out vec4 fragColor;

#define AMOUNT  0.6   // [0, 3, 0.05] sharpening strength
#define RADIUS  1.0   // [0.3, 3, 0.1] sample radius in pixels
#define CLAMP   0.35  // [0, 1, 0.01] max change per channel (halo guard)

void main() {
    vec2 t = RADIUS / uResolution;
    vec3 c  = texture(uMain, vUV).rgb;

    // 4-tap cross blur = local average
    vec3 blur = texture(uMain, vUV + vec2(t.x, 0.0)).rgb
              + texture(uMain, vUV - vec2(t.x, 0.0)).rgb
              + texture(uMain, vUV + vec2(0.0, t.y)).rgb
              + texture(uMain, vUV - vec2(0.0, t.y)).rgb;
    blur *= 0.25;

    vec3 diff = (c - blur) * AMOUNT;
    diff = clamp(diff, -CLAMP, CLAMP);   // stop bright halos on hard edges
    fragColor = vec4(clamp(c + diff, 0.0, 1.0), 1.0);
}
