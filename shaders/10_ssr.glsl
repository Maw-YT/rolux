#version 330 core
// 10_ssr.glsl — screen-space reflections for RoLux (monocular depth).
//
// View-space positions + normals are derived from uDepthF with the same
// pseudo-projection as the normal pass. Marches a reflected ray in eye space
// and samples uScene at the hit UV.
//
// Slider annotations for the RoLux GUI:  // [min, max, step] description

uniform sampler2D uScene;
uniform sampler2D uDepthF;
uniform sampler2D uNormal;
uniform vec2  uResolution;
uniform float uTime;

in  vec2 vUV;
out vec4 fragColor;

// ------------------------- tunables (hot-reload) -------------------------
#define FOV_DEG           70.0   // [30, 120, 1] Roblox vertical FOV
#define Z_NEAR            0.25   // [0.05, 2.0, 0.01] pseudo near plane
#define Z_FAR             16.0   // [2.0, 40.0, 0.5] pseudo far plane
#define DEPTH_NEAR_IS_ONE 1      // [0, 1, 1] DA-V2 disparity: 1 = near

#define MAX_STEPS         48     // [8, 160, 1] linear march steps
#define BINARY_STEPS      5      // [0, 12, 1] binary refinement steps
#define RAY_MAX_DIST      14.0   // [1, 32, 0.5] max eye-space march length
#define STEP_JITTER       0.85   // [0, 1, 0.05] per-pixel dither vs banding
#define THICKNESS         0.35   // [0.05, 4.0, 0.05] hit tolerance (eye-space)
#define DEPTH_BIAS        0.012  // [0.0, 0.2, 0.005] self-intersection guard

#define INTENSITY         1.05   // [0, 2, 0.05] reflection strength
#define FRESNEL_POW       2.8    // [0.5, 8, 0.1] grazing falloff
#define FRESNEL_MIN       0.08   // [0, 1, 0.01] reflectivity at normal incidence

#define REFLECT_MASK      0      // [0, 1, 1] 0: everything  1: up-facing only
#define UP_SIGN           1.0    // [-1, 1, 2] flip if it lands on ceilings
#define UP_MIN            0.25   // [0, 1, 0.02] min N.y to count as floor
#define EDGE_FADE         0.08   // [0, 0.4, 0.01] screen-edge fade width

#define ROUGHNESS         0.08   // [0, 1, 0.02] 0 = mirror, higher = blurry
#define BLUR_TAPS         3      // [0, 12, 1] roughness blur ring taps

#define USE_NORMAL_TEX    1      // [0, 1, 1] 0: derive from depth  1: uNormal texture

#define DEBUG_VIEW        0      // [0, 2, 1] 0: off  1: reflection only  2: mask
// -------------------------------------------------------------------------

float depthRaw(vec2 uv) {
    return textureLod(uDepthF, clamp(uv, vec2(0.0), vec2(1.0)), 0.0).r;
}

float linDepth(vec2 uv) {
    float d = depthRaw(uv);
#if DEPTH_NEAR_IS_ONE
    d = 1.0 - d;
#endif
    return mix(Z_NEAR, Z_FAR, clamp(d, 0.0, 1.0));
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

vec2 uvFromViewPos(vec3 p, out float lz) {
    vec2 th = tanHalf();
    lz = max(-p.z, 1e-4);
    vec2 ndc = vec2(p.x / (th.x * lz), p.y / (th.y * lz));
    return ndc * 0.5 + 0.5;
}

vec3 viewNormalFromDepth(vec2 uv) {
    vec2 ts = 1.0 / max(uResolution, vec2(1.0));
    float zc = depthRaw(uv);
    float thr = max(zc * 0.025, 0.01);
    float zr = abs(depthRaw(uv + vec2(ts.x, 0.0)) - zc) > thr ? zc : depthRaw(uv + vec2(ts.x, 0.0));
    float zu = abs(depthRaw(uv + vec2(0.0, ts.y)) - zc) > thr ? zc : depthRaw(uv + vec2(0.0, ts.y));
    float lz = linDepth(uv);
    vec3 p  = viewPosFromUV(uv, lz);
    vec3 pR = viewPosFromUV(uv + vec2(ts.x, 0.0), linDepthRaw(zr));
    vec3 pU = viewPosFromUV(uv + vec2(0.0, ts.y), linDepthRaw(zu));
    vec3 n  = normalize(cross(pR - p, pU - p));
    if (dot(n, normalize(-p)) < 0.0) n = -n;
    return n;
}

vec3 getNormal(vec2 uv) {
#if USE_NORMAL_TEX == 1
    vec3 n = normalize(texture(uNormal, uv).xyz * 2.0 - 1.0);
    vec3 p = viewPosFromUV(uv, linDepth(uv));
    if (dot(n, normalize(-p)) < 0.0) n = -n;
    return n;
#else
    return viewNormalFromDepth(uv);
#endif
}

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p.y) * p3.z);
}

void main() {
    vec3 base = texture(uScene, vUV).rgb;

    float lz = linDepth(vUV);
    vec3  P  = viewPosFromUV(vUV, lz);
    vec3  N  = getNormal(vUV);
    vec3  V  = normalize(P);              // camera -> surface (incident)
    vec3  R  = reflect(V, N);             // reflected direction in view space

    float mask = 1.0;
#if REFLECT_MASK == 1
    mask = smoothstep(UP_MIN, UP_MIN + 0.25, N.y * UP_SIGN);
#endif
    float ndv = max(dot(normalize(-P), N), 0.0);
    float fres = FRESNEL_MIN + (1.0 - FRESNEL_MIN) * pow(1.0 - ndv, FRESNEL_POW);
    float refl = mask * fres;

#if DEBUG_VIEW == 2
    fragColor = vec4(refl, mask, fres, 1.0);
    return;
#endif

    if (refl < 0.001 && DEBUG_VIEW == 0) { fragColor = vec4(base, 1.0); return; }

    // Adaptive thickness: scale with hit distance so far hits aren't rejected.
    float stepLen = RAY_MAX_DIST / float(MAX_STEPS);
    float jitter  = STEP_JITTER * hash12(vUV * uResolution + uTime);
    vec3  marchP  = P + N * DEPTH_BIAS + R * stepLen * jitter;
    vec3  prevP   = marchP;

    bool  hit   = false;
    vec2  hitUV = vUV;

    for (int i = 0; i < MAX_STEPS; i++) {
        marchP += R * stepLen;
        float rayLz;
        vec2  uv = uvFromViewPos(marchP, rayLz);
        if (any(lessThan(uv, vec2(0.0))) || any(greaterThan(uv, vec2(1.0)))) break;
        if (rayLz <= Z_NEAR * 0.5) break;

        float sceneLz = linDepth(uv);
        float diff = rayLz - sceneLz;
        float thick = THICKNESS * (1.0 + rayLz * 0.08);

        if (diff > DEPTH_BIAS && diff < thick) {
            hit = true;
            hitUV = uv;
            vec3 a = prevP, b = marchP;
            for (int j = 0; j < BINARY_STEPS; j++) {
                vec3 m = (a + b) * 0.5;
                float mlz;
                vec2 muv = uvFromViewPos(m, mlz);
                if (mlz - linDepth(muv) > DEPTH_BIAS) { b = m; hitUV = muv; }
                else                                  { a = m; }
            }
            break;
        }
        prevP = marchP;
    }

    vec3 rcol = hit ? texture(uScene, hitUV).rgb : base;

#if BLUR_TAPS > 0
    if (ROUGHNESS > 0.001 && hit) {
        vec2 rad = vec2(ROUGHNESS * 0.025);
        vec3 acc = rcol;
        float wsum = 1.0;
        for (int k = 0; k < BLUR_TAPS; k++) {
            float ang = 6.2831853 * (float(k) + 0.5) / float(BLUR_TAPS);
            acc += texture(uScene, hitUV + vec2(cos(ang), sin(ang)) * rad).rgb;
            wsum += 1.0;
        }
        rcol = acc / wsum;
    }
#endif

#if DEBUG_VIEW == 1
    fragColor = vec4(hit ? rcol : vec3(0.0), 1.0);
    return;
#endif

    if (!hit) { fragColor = vec4(base, 1.0); return; }

    vec2 fe = smoothstep(vec2(0.0), vec2(EDGE_FADE), hitUV)
            * smoothstep(vec2(0.0), vec2(EDGE_FADE), 1.0 - hitUV);
    float edge = fe.x * fe.y;
    float distFade = 1.0 - clamp(length(marchP - P) / RAY_MAX_DIST, 0.0, 1.0);

    float k = clamp(INTENSITY * refl * edge * distFade, 0.0, 1.0);
    fragColor = vec4(mix(base, rcol, k), 1.0);
}
