#version 330 core
// 10_ssr.glsl — screen-space reflections for RoLux (monocular depth).
//
// Reads uScene (Roblox color + reflection source), uDepthF (relative depth),
// uNormal (view-space normals). Outputs the scene with SSR composited onto
// reflective surfaces. Self-contained: it samples uScene directly, so chain
// position / other debug passes don't affect its result.
//
// DepthAnythingV2 depth has no metric scale, so eye-space here is a *pseudo*
// space shaped by Z_NEAR/Z_FAR/FOV_DEG. Tune those first. If you see nothing,
// set DEBUG_VIEW to 1 to render only what the raymarch actually hits.
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

#define MAX_STEPS         56     // [8, 160, 1] linear march steps
#define BINARY_STEPS      6      // [0, 12, 1] binary refinement steps
#define RAY_MAX_DIST      12.0   // [1, 32, 0.5] max eye-space march length
#define STEP_JITTER       1.0    // [0, 1, 0.05] per-pixel dither vs banding
#define THICKNESS         0.9    // [0.05, 4.0, 0.05] hit tolerance (eye-space)
#define DEPTH_BIAS        0.02   // [0.0, 0.2, 0.005] self-intersection guard

#define INTENSITY         1.0    // [0, 2, 0.05] reflection strength
#define FRESNEL_POW       3.0    // [0.5, 8, 0.1] grazing falloff
#define FRESNEL_MIN       0.12   // [0, 1, 0.01] reflectivity at normal incidence

#define REFLECT_MASK      0      // [0, 1, 1] 0: everything  1: up-facing only
#define UP_SIGN           1.0    // [-1, 1, 2] flip if it lands on ceilings
#define UP_MIN            0.3   // [0, 1, 0.02] min N.y to count as floor
#define EDGE_FADE         0.12   // [0, 0.4, 0.01] screen-edge fade width

#define ROUGHNESS         0.12   // [0, 1, 0.02] 0 = mirror, higher = blurry
#define BLUR_TAPS         4      // [0, 12, 1] roughness blur ring taps

#define DEBUG_VIEW        0      // [0, 2, 1] 0: off  1: reflection only  2: mask
// -------------------------------------------------------------------------

float depthRaw(vec2 uv) {
    return textureLod(uDepthF, clamp(uv, 0.0, 1.0), 0.0).r;
}

// relative depth -> pseudo eye-space distance (positive, grows with distance)
float linDepth(vec2 uv) {
    float d = depthRaw(uv);
#if DEPTH_NEAR_IS_ONE
    d = 1.0 - d;                 // 0 = near, 1 = far
#endif
    return mix(Z_NEAR, Z_FAR, clamp(d, 0.0, 1.0));
}

vec3 decodeNormal(vec2 uv) {
    return normalize(texture(uNormal, uv).xyz * 2.0 - 1.0);
}

vec2 tanHalf() {
    float tv = tan(radians(FOV_DEG * 0.5));
    float aspect = uResolution.x / max(uResolution.y, 1.0);
    return vec2(tv * aspect, tv);
}

// camera at origin, looking down -Z
vec3 viewPosFromUV(vec2 uv, float lz) {
    vec2 th  = tanHalf();
    vec2 ndc = uv * 2.0 - 1.0;
    return vec3(ndc.x * th.x, ndc.y * th.y, -1.0) * lz;
}

vec2 uvFromViewPos(vec3 p, out float lz) {
    vec2 th = tanHalf();
    lz = -p.z;
    vec2 ndc = vec2(p.x / (th.x * lz), p.y / (th.y * lz));
    return ndc * 0.5 + 0.5;
}

float hash12(vec2 p) {
    vec3 p3 = fract(vec3(p.xyx) * 0.1031);
    p3 += dot(p3, p3.yzx + 33.33);
    return fract((p3.x + p3.y) * p3.z);
}

void main() {
    vec3 base = texture(uScene, vUV).rgb;

    float lz = linDepth(vUV);
    vec3  P  = viewPosFromUV(vUV, lz);   // eye-space fragment position
    vec3  N  = decodeNormal(vUV);        // eye-space normal (+z toward camera)
    vec3  V  = normalize(P);             // camera -> fragment
    vec3  R  = reflect(V, N);            // reflected direction

    // reflectivity gate: surface mask * fresnel
    float mask = 1.0;
#if REFLECT_MASK == 1
    mask = smoothstep(UP_MIN, UP_MIN + 0.25, N.y * UP_SIGN);
#endif
    float fres = FRESNEL_MIN + (1.0 - FRESNEL_MIN)
               * pow(clamp(1.0 + dot(V, N), 0.0, 1.0), FRESNEL_POW);
    float refl = mask * fres;

#if DEBUG_VIEW == 2
    fragColor = vec4(refl, mask, fres, 1.0);
    return;
#endif

    if (refl < 0.002 && DEBUG_VIEW == 0) { fragColor = vec4(base, 1.0); return; }

    // linear march
    float stepLen = RAY_MAX_DIST / float(MAX_STEPS);
    float jitter  = STEP_JITTER * hash12(vUV * uResolution + uTime);
    vec3  marchP  = P + N * DEPTH_BIAS + R * stepLen * jitter;
    vec3  prevP   = marchP;

    bool hit = false;
    vec2 hitUV = vUV;

    for (int i = 0; i < MAX_STEPS; i++) {
        marchP += R * stepLen;
        float rayLz;
        vec2  uv = uvFromViewPos(marchP, rayLz);
        if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0) break;
        if (rayLz <= 0.0) break;

        float diff = rayLz - linDepth(uv);   // >0: ray behind the scene surface
        if (diff > DEPTH_BIAS && diff < THICKNESS) {
            hit = true; hitUV = uv;
            vec3 a = prevP, b = marchP;       // binary refine
            for (int j = 0; j < BINARY_STEPS; j++) {
                vec3 m = (a + b) * 0.5;
                float mlz; vec2 muv = uvFromViewPos(m, mlz);
                if (mlz - linDepth(muv) > DEPTH_BIAS) { b = m; hitUV = muv; }
                else                                  { a = m; }
            }
            break;
        }
        prevP = marchP;
    }

    // roughness blur around the hit
    vec3 rcol = texture(uScene, hitUV).rgb;
#if BLUR_TAPS > 0
    if (ROUGHNESS > 0.001 && hit) {
        vec2 rad = vec2(ROUGHNESS * 0.03);
        vec3 acc = rcol; float wsum = 1.0;
        for (int k = 0; k < BLUR_TAPS; k++) {
            float ang = 6.2831853 * float(k) / float(BLUR_TAPS);
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

    // fades: screen edge + ray distance
    vec2 fe = smoothstep(vec2(0.0), vec2(EDGE_FADE), hitUV)
            * smoothstep(vec2(0.0), vec2(EDGE_FADE), 1.0 - hitUV);
    float edge = fe.x * fe.y;
    float distFade = 1.0 - clamp(length(P - marchP) / RAY_MAX_DIST, 0.0, 1.0);

    float k = clamp(INTENSITY * refl * edge * distFade, 0.0, 1.0);
    fragColor = vec4(mix(base, rcol, k), 1.0);
}
