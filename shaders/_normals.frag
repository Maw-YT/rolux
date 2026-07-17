#version 330 core
// Depth -> view-space normals (RoLux). Hot-reloaded from disk.
//
// Reconstructs view-space positions from uDepthF (same pseudo-projection as SSR)
// and takes a cross product — NOT a 2D screen gradient.
//
// Soft linear upscales create a multi-pixel depth ramp at silhouettes; normals
// on that ramp look like a thick colorful outline. We sharpen depth toward the
// nearer of {local min, max} so the ramp collapses to ~1px, then estimate
// normals with central (or same-surface one-sided) differences — preserving
// real surface tilt instead of flattening to camera-facing.
//
// Keep FOV_DEG / Z_NEAR / Z_FAR / DEPTH_NEAR_IS_ONE in sync with 10_ssr.glsl.
// Slider annotations:  // [min, max, step] description

uniform sampler2D uDepthF;
uniform vec2 uResolution;
uniform float uStrength;  // tilt scale: 1 = as estimated, <1 flattens, >1 exaggerates
in vec2 vUV;
out vec4 fragColor;

#define FOV_DEG           70.0
#define Z_NEAR            0.25
#define Z_FAR             16.0
#define DEPTH_NEAR_IS_ONE 1

#define EDGE_THR_REL      0.02   // [0.005, 0.1, 0.005] relative depth edge threshold
#define EDGE_THR_ABS      0.008  // [0.001, 0.05, 0.001] absolute floor for edge reject
#define SHARPEN_SCALE     1.0    // [0.5, 2.5, 0.05] neighborhood range vs thr to snap ramp
#define TAP_SOFT          0.15   // [0, 0.5, 0.05] light same-surface blur for flats only

float depthRaw(vec2 uv) {
    return textureLod(uDepthF, clamp(uv, vec2(0.0), vec2(1.0)), 0.0).r;
}

float edgeThr(float zc) {
    return max(zc * EDGE_THR_REL, EDGE_THR_ABS);
}

bool sameSurface(float z, float zc) {
    return abs(z - zc) <= edgeThr(zc);
}

// Collapse soft upscale ramps: if the 3×3 span is an edge, snap to the nearer
// of local min/max so silhouette normals aren't computed on the blend zone.
float depthSharp(vec2 uv) {
    vec2 ts = 1.0 / max(uResolution, vec2(1.0));
    float zc = depthRaw(uv);
    float zmin = zc;
    float zmax = zc;
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            float z = depthRaw(uv + vec2(float(i), float(j)) * ts);
            zmin = min(zmin, z);
            zmax = max(zmax, z);
        }
    }
    float thr = edgeThr(zc) * SHARPEN_SCALE;
    if ((zmax - zmin) > thr) {
        return (abs(zc - zmin) <= abs(zc - zmax)) ? zmin : zmax;
    }

    // Tiny same-surface average — flats only (span already below thr).
    if (TAP_SOFT <= 1e-4) {
        return zc;
    }
    float sum = zc;
    float wsum = 1.0;
    for (int j = -1; j <= 1; ++j) {
        for (int i = -1; i <= 1; ++i) {
            if (i == 0 && j == 0) continue;
            float z = depthRaw(uv + vec2(float(i), float(j)) * ts);
            if (!sameSurface(z, zc)) continue;
            float w = TAP_SOFT;
            sum += z * w;
            wsum += w;
        }
    }
    return sum / max(wsum, 1e-5);
}

float linDepthRaw(float d) {
#if DEPTH_NEAR_IS_ONE
    d = 1.0 - d;
#endif
    return mix(Z_NEAR, Z_FAR, clamp(d, 0.0, 1.0));
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

// Prefer central difference; if one side crosses a discontinuity, use the
// remaining same-surface side (keeps object tilt at the rim). Never substitute
// center depth (flat halo) and never force camera-facing for partial edges.
bool axisDelta(vec2 uv, vec2 axis, float zc, out vec3 dAxis) {
    vec2 ts = axis / max(uResolution, vec2(1.0));
    float zPos = depthSharp(uv + ts);
    float zNeg = depthSharp(uv - ts);
    bool vPos = sameSurface(zPos, zc);
    bool vNeg = sameSurface(zNeg, zc);

    vec3 p = viewPosFromUV(uv, linDepthRaw(zc));
    if (vPos && vNeg) {
        dAxis = viewPosFromUV(uv + ts, linDepthRaw(zPos))
              - viewPosFromUV(uv - ts, linDepthRaw(zNeg));
        return true;
    }
    if (vPos) {
        dAxis = viewPosFromUV(uv + ts, linDepthRaw(zPos)) - p;
        return true;
    }
    if (vNeg) {
        dAxis = p - viewPosFromUV(uv - ts, linDepthRaw(zNeg));
        return true;
    }
    dAxis = vec3(0.0);
    return false;
}

void main() {
    float zc = depthSharp(vUV);
    vec3 p = viewPosFromUV(vUV, linDepthRaw(zc));
    vec3 face = normalize(-p);

    vec3 dX, dY;
    bool hasX = axisDelta(vUV, vec2(1.0, 0.0), zc, dX);
    bool hasY = axisDelta(vUV, vec2(0.0, 1.0), zc, dY);

    vec3 n;
    if (hasX && hasY) {
        n = cross(dX, dY);
        float len2 = dot(n, n);
        n = (len2 < 1e-12) ? face : n * inversesqrt(len2);
    } else if (hasX) {
        // Single-axis: build a normal from dX × approximate up in screen space.
        vec3 up = abs(face.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
        n = cross(dX, cross(up, dX));
        float len2 = dot(n, n);
        n = (len2 < 1e-12) ? face : n * inversesqrt(len2);
    } else if (hasY) {
        vec3 rt = abs(face.x) < 0.99 ? vec3(1.0, 0.0, 0.0) : vec3(0.0, 1.0, 0.0);
        n = cross(cross(dY, rt), dY);
        float len2 = dot(n, n);
        n = (len2 < 1e-12) ? face : n * inversesqrt(len2);
    } else {
        n = face;
    }

    if (dot(n, face) < 0.0) n = -n;

    // uStrength: 1 = raw estimate. Default path should leave tilt alone.
    float s = max(uStrength, 0.01);
    if (abs(s - 1.0) > 1e-3) {
        n = normalize(mix(face, n, s));
        if (dot(n, face) < 0.0) n = -n;
    }

    fragColor = vec4(clamp(n * 0.5 + 0.5, 0.0, 1.0), 1.0);
}
