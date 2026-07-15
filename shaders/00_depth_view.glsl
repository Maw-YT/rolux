#version 330 core
// 00_depth_view.glsl — show monocular depth as grayscale overlay.

uniform sampler2D uDepth;

in vec2 vUV;
out vec4 fragColor;

void main() {
    float d = texture(uDepth, vUV).r;
    fragColor = vec4(vec3(d), 1.0);
}
