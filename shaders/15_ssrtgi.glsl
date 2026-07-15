#version 330 core
// 15_ssrtgi.glsl — screen-space ray-traced global illumination (RTGI-style).
//
// For every pixel it fires cosine-weighted rays into the hemisphere around the
// depth-derived normal and marches them against the depth buffer:
//   * a hit contributes AMBIENT OCCLUSION (contact shadowing) and
//   * ONE-BOUNCE INDIRECT DIFFUSE (color bleed from the hit surface).
// Misses are treated as unoccluded (open sky / ambient).
//
// Reads uMain (chains after SSR — used as base + bounce color), uDepthF, uNormal.
// Depth is monocular/relative, so eye-space is a *pseudo* space shaped by
// Z_NEAR/Z_FAR/FOV_DEG — tune those to match SSR. Use DEBUG_VIEW to inspect.
//
// Slider annotations for the RoLux GUI:  // [min, max, step] description

uniform sampler2D uMain;
uniform sampler2D uDepthF;
uniform sampler2D uNormal;
uniform vec2  uResolution;
uniform float uTime;

in  vec2 vUV;
out vec4 fragColor;

// ------------------------- depth / projection ----------------------------
#define FOV_DEG           70.0   // [30, 120, 1] Roblox vertical FOV
#define Z_NEAR            0.25   // [0.05, 2.0, 0.01] pseudo near plane
#define Z_FAR             16.0   // [2.0, 40.0, 0.5] pseudo far plane
#define DEPTH_NEAR_IS_ONE 1      // [0, 1, 1] DA-V2 disparity: 1 = near

// ------------------------- ray tracing -----------------------------------
#define RAYS              4      // [1, 8, 1] hemisphere rays per pixel
#define STEPS             14     // [4, 32, 1] march steps per ray
#define RAY_RADIUS        3.5    // [0.2, 12, 0.1] eye-space gather radius
#define STEP_GROWTH       1.25   // [1, 1.6, 0.01] step size multiplier
#define THICKNESS         0.9    // [0.05, 4, 0.05] occluder thickness
#define BIAS              0.03   // [0, 0.3, 0.005] normal bias (self-occ guard)
#define FALLOFF           1.5    // [0.2, 4, 0.1] distance falloff power

// ------------------------- look ------------------------------------------
#define AO_AMOUNT         1.0    // [0, 3, 0.05] occlusion strength
#define AO_POWER          1.4    // [0.5, 4, 0.1] occlusion contrast
#define GI_AMOUNT         1.2    // [0, 4, 0.05] indirect bounce strength
#define GI_SAT            1.3    // [0, 3, 0.05] bounce color saturation
#define AMBIENT           1.0    // [0, 2, 0.05] base (unlit) light kept

#define DEBUG_VIEW        0      // [0, 3, 1] 0:off 1:AO 2:GI 3:normal
// -------------------------------------------------------------------------

const float TWO_PI = 6.2831853;

float depthRaw(vec2 uv) { return textureLod(uDepthF, clamp(uv, 0.0, 1.0), 0.0).r; }

float linDepth(vec2 uv) {
    float d = depthRaw(uv);
#if DEPTH_NEAR_IS_ONE
    d = 1.0 - d;
#endif
    return mix(Z_NEAR, Z_FAR, clamp(d, 0.0, 1.0));
}

vec3 decodeNormal(vec2 uv) { return normalize(texture(uNormal, uv).xyz * 2.0 - 1.0); }

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
    lz = -p.z;
    return vec2(p.x / (th.x * lz), p.y / (th.y * lz)) * 0.5 + 0.5;
}

float hash13(vec3 p) {
    p = fract(p * 0.1031);
    p += dot(p, p.zyx + 31.32);
    return fract((p.x + p.y) * p.z);
}

void main() {
    vec3 base = texture(uMain, vUV).rgb;
    float lz = linDepth(vUV);
    vec3  P  = viewPosFromUV(vUV, lz);
    vec3  N  = decodeNormal(vUV);

#if DEBUG_VIEW == 3
    fragColor = vec4(N * 0.5 + 0.5, 1.0);
    return;
#endif

    // tangent basis around the normal
    vec3 up = abs(N.y) < 0.99 ? vec3(0.0, 1.0, 0.0) : vec3(1.0, 0.0, 0.0);
    vec3 T  = normalize(cross(up, N));
    vec3 B  = cross(N, T);

    float occ = 0.0;
    vec3  gi  = vec3(0.0);
    float seed = uTime * 0.37;

    for (int r = 0; r < RAYS; r++) {
        float r1 = hash13(vec3(vUV * uResolution, float(r) + seed));
        float r2 = hash13(vec3(vUV * uResolution + 7.1, float(r) * 1.7 + seed));

        // cosine-weighted hemisphere direction
        float phi = TWO_PI * r1;
        float ct  = sqrt(1.0 - r2);
        float st  = sqrt(r2);
        vec3  dir = T * (cos(phi) * st) + B * (sin(phi) * st) + N * ct;

        float t    = RAY_RADIUS / float(STEPS);
        float step = t;
        for (int s = 0; s < STEPS; s++) {
            vec3 marchP = P + N * BIAS + dir * t;
            float rayLz;
            vec2 uv = uvFromViewPos(marchP, rayLz);
            if (uv.x < 0.0 || uv.x > 1.0 || uv.y < 0.0 || uv.y > 1.0 || rayLz <= 0.0) break;

            float diff = rayLz - linDepth(uv);       // >0: ray behind a surface
            if (diff > BIAS && diff < THICKNESS) {
                float fall = pow(1.0 - clamp(t / RAY_RADIUS, 0.0, 1.0), FALLOFF);
                occ += fall;
                gi  += texture(uMain, uv).rgb * fall;
                break;
            }
            step *= STEP_GROWTH;
            t += step;
        }
    }

    float inv = 1.0 / float(RAYS);
    occ *= inv;
    gi  *= inv;

    // ambient occlusion
    float ao = clamp(1.0 - occ * AO_AMOUNT, 0.0, 1.0);
    ao = pow(ao, AO_POWER);

    // indirect diffuse (color bleed), saturation-shaped
    float gl = dot(gi, vec3(0.2126, 0.7152, 0.0722));
    gi = mix(vec3(gl), gi, GI_SAT) * GI_AMOUNT;

#if DEBUG_VIEW == 1
    fragColor = vec4(vec3(ao), 1.0);
    return;
#endif
#if DEBUG_VIEW == 2
    fragColor = vec4(clamp(gi, 0.0, 1.0), 1.0);
    return;
#endif

    // receiver-modulated: darken by AO, add bounce tinted by the receiver
    vec3 outc = base * (AMBIENT * ao) + base * gi;
    fragColor = vec4(clamp(outc, 0.0, 1.0), 1.0);
}
