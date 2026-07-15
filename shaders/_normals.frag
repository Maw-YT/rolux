#version 330 core
// Depth -> view-space normals (RoLux). Hot-reloaded from disk.
uniform sampler2D uDepthF;
uniform vec2 uResolution;
uniform float uStrength;
in vec2 vUV;
out vec4 fragColor;

float sampleD(vec2 uv) {
    return textureLod(uDepthF, clamp(uv, vec2(0.0), vec2(1.0)), 0.0).r;
}

// Bilateral tap: reject samples across a real depth discontinuity (silhouette
// edges) so normals don't bleed across objects. Threshold scales with the
// center depth itself instead of a fixed constant, since a flat threshold
// doesn't track well across near/far parts of the scene or across different
// depth-model output ranges.
float tap(vec2 uv, float zc, float thr) {
    float z = sampleD(uv);
    return abs(z - zc) > thr ? zc : z;
}

void main() {
    vec2 ts = vec2(textureSize(uDepthF, 0));
    vec2 t = 1.0 / max(ts, vec2(1.0));

    float zc = sampleD(vUV);
    float thr = max(zc * 0.02, 0.01);

    float zx1 = tap(vUV + vec2(t.x, 0.0), zc, thr);
    float zx0 = tap(vUV - vec2(t.x, 0.0), zc, thr);
    float zy1 = tap(vUV + vec2(0.0, t.y), zc, thr);
    float zy0 = tap(vUV - vec2(0.0, t.y), zc, thr);

    // Plain central difference. No coarse-minus-fine cancellation: a smooth
    // linear depth ramp (a sloped surface) IS the signal we want, not noise
    // to be filtered out.
    float dx = (zx1 - zx0) * 0.5;
    float dy = (zy1 - zy0) * 0.5;

    float s = 56.0 * max(uStrength, 0.5);
    vec3 n = normalize(vec3(-dx * s, dy * s, 1.0));
    n.z = max(n.z, 0.05);
    n = normalize(n);
    fragColor = vec4(clamp(n * 0.5 + 0.5, 0.0, 1.0), 1.0);
}
