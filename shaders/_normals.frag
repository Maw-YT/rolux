#version 330 core
// Depth -> view-space normals (RoLux). Hot-reloaded from disk.
//
// Reconstructs view-space positions from uDepthF (same pseudo-projection as SSR)
// and takes a cross product — NOT a 2D screen gradient. Neighbor taps reject
// depth discontinuities so normals don't bleed across silhouettes.
//
// Keep FOV_DEG / Z_NEAR / Z_FAR / DEPTH_NEAR_IS_ONE in sync with 10_ssr.glsl.

uniform sampler2D uDepthF;
uniform vec2 uResolution;
uniform float uStrength;
in vec2 vUV;
out vec4 fragColor;

#define FOV_DEG           70.0
#define Z_NEAR            0.25
#define Z_FAR             16.0
#define DEPTH_NEAR_IS_ONE 1

float depthRaw(vec2 uv) {
    return textureLod(uDepthF, clamp(uv, vec2(0.0), vec2(1.0)), 0.0).r;
}

float linDepthRaw(float d) {
#if DEPTH_NEAR_IS_ONE
    d = 1.0 - d;
#endif
    return mix(Z_NEAR, Z_FAR, clamp(d, 0.0, 1.0));
}

float linDepth(vec2 uv) {
    return linDepthRaw(depthRaw(uv));
}

// Reject neighbor samples across a depth edge (same idea as the old bilateral tap).
float tapRaw(vec2 uv, float zc) {
    float z = depthRaw(uv);
    float thr = max(zc * 0.025, 0.01);
    return abs(z - zc) > thr ? zc : z;
}

vec2 tanHalf() {
    float tv = tan(radians(FOV_DEG * 0.5));
    return vec2(tv * uResolution.x / max(uResolution.y, 1.0), tv);
}

vec3 viewPosFromUV(vec2 uv, float lz) {
    vec2 ndc = uv * 2.0 - 1.0;
    vec2 th  = tanHalf();
    return vec3(ndc.x * th.x, ndc.y * th.y, -1.0) * lz;
}

vec3 viewNormal(vec2 uv) {
    vec2 ts = 1.0 / max(uResolution, vec2(1.0));
    float zc = depthRaw(uv);
    float zr = tapRaw(uv + vec2(ts.x, 0.0), zc);
    float zu = tapRaw(uv + vec2(0.0, ts.y), zc);

    vec3 p  = viewPosFromUV(uv, linDepthRaw(zc));
    vec3 pR = viewPosFromUV(uv + vec2(ts.x, 0.0), linDepthRaw(zr));
    vec3 pU = viewPosFromUV(uv + vec2(0.0, ts.y), linDepthRaw(zu));
    return normalize(cross(pR - p, pU - p));
}

void main() {
    vec3 n = viewNormal(vUV);
    vec3 p = viewPosFromUV(vUV, linDepth(vUV));
    if (dot(n, normalize(-p)) < 0.0) n = -n;
    fragColor = vec4(clamp(n * 0.5 + 0.5, 0.0, 1.0), 1.0);
}
