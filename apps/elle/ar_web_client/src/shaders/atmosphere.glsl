/**
 * Atmospheric Effects Shaders - Environmental and cinematic effects
 *
 * Implements:
 * - Volumetric fog (exponential, height-based)
 * - God rays (volumetric lighting)
 * - Rain/snow particle effects
 * - Color grading (cinematic LUTs)
 * - Depth of field (bokeh)
 *
 * Used for cinematic atmospheric effects in dream scenes.
 */

// ============================================================================
// VOLUMETRIC FOG SHADER
// ============================================================================

// Fragment shader - Volumetric fog with noise
#version 300 es
precision highp float;

uniform vec3 fogColor;
uniform float fogDensity;
uniform float time;
uniform sampler2D sceneTexture;
uniform sampler2D depthTexture;

in vec2 vUv;
out vec4 fragColor;

// Noise function
float noise(vec3 p) {
    vec3 i = floor(p);
    vec3 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);

    float n = i.x + i.y * 57.0 + 113.0 * i.z;
    return mix(
        mix(mix(fract(sin(n) * 43758.5453),
                fract(sin(n + 1.0) * 43758.5453), f.x),
            mix(fract(sin(n + 57.0) * 43758.5453),
                fract(sin(n + 58.0) * 43758.5453), f.x), f.y),
        mix(mix(fract(sin(n + 113.0) * 43758.5453),
                fract(sin(n + 114.0) * 43758.5453), f.x),
            mix(fract(sin(n + 170.0) * 43758.5453),
                fract(sin(n + 171.0) * 43758.5453), f.x), f.y), f.z);
}

void main() {
    vec4 sceneColor = texture(sceneTexture, vUv);
    float depth = texture(depthTexture, vUv).r;

    // Reconstruct world position from depth
    float linearDepth = depth * 100.0;  // Far plane = 100

    // Animated noise
    vec3 noiseCoord = vec3(vUv * 10.0, time * 0.1);
    float noiseValue = noise(noiseCoord) * 0.5 + 0.5;

    // Exponential fog with noise modulation
    float fogAmount = 1.0 - exp(-fogDensity * linearDepth * (0.8 + noiseValue * 0.4));

    // Height-based fog (thicker near ground)
    float heightFog = smoothstep(0.0, 0.3, vUv.y);
    fogAmount *= (0.5 + heightFog * 0.5);

    // Mix scene with fog
    vec3 finalColor = mix(sceneColor.rgb, fogColor, fogAmount);

    fragColor = vec4(finalColor, 1.0);
}


// ============================================================================
// GOD RAYS (VOLUMETRIC LIGHTING)
// ============================================================================

// Fragment shader - God rays from light source
#version 300 es
precision highp float;

uniform sampler2D sceneTexture;
uniform sampler2D depthTexture;
uniform vec2 lightPosition;  // Screen space light position
uniform float exposure;
uniform float decay;
uniform float density;
uniform float weight;

in vec2 vUv;
out vec4 fragColor;

const int NUM_SAMPLES = 100;

void main() {
    vec4 sceneColor = texture(sceneTexture, vUv);

    // Ray marching from pixel to light
    vec2 deltaTextCoord = vUv - lightPosition;
    deltaTextCoord *= 1.0 / float(NUM_SAMPLES) * density;

    vec2 textCoord = vUv;
    float illuminationDecay = 1.0;
    vec3 godRays = vec3(0.0);

    for (int i = 0; i < NUM_SAMPLES; i++) {
        textCoord -= deltaTextCoord;
        vec4 sample = texture(sceneTexture, textCoord);

        sample *= illuminationDecay * weight;
        godRays += sample.rgb;

        illuminationDecay *= decay;
    }

    godRays *= exposure;

    // Additive blend with scene
    vec3 finalColor = sceneColor.rgb + godRays;

    fragColor = vec4(finalColor, 1.0);
}


// ============================================================================
// RAIN STREAKS SHADER
// ============================================================================

// Fragment shader - Screen-space rain streaks
#version 300 es
precision highp float;

uniform sampler2D sceneTexture;
uniform float time;
uniform float intensity;

in vec2 vUv;
out vec4 fragColor;

float noise(vec2 p) {
    return fract(sin(dot(p, vec2(12.9898, 78.233))) * 43758.5453);
}

void main() {
    vec4 sceneColor = texture(sceneTexture, vUv);

    // Rain streak coordinates
    vec2 rainCoord = vUv * vec2(20.0, 40.0);
    rainCoord.y += time * 5.0;  // Fall speed

    // Random rain drops
    float rain = 0.0;
    for (int i = 0; i < 5; i++) {
        vec2 offset = vec2(float(i) * 3.7, 0.0);
        vec2 coord = rainCoord + offset;

        float n = noise(floor(coord));
        if (n > 0.95) {  // Sparse rain
            float streak = smoothstep(0.0, 0.1, fract(coord.y)) * smoothstep(1.0, 0.9, fract(coord.y));
            float width = smoothstep(0.0, 0.05, abs(fract(coord.x) - 0.5));
            rain += streak * width * intensity;
        }
    }

    // Additive rain
    vec3 finalColor = sceneColor.rgb + vec3(rain * 0.5);

    fragColor = vec4(finalColor, 1.0);
}


// ============================================================================
// CINEMATIC COLOR GRADING
// ============================================================================

// Fragment shader - Film-style color grading
#version 300 es
precision highp float;

uniform sampler2D sceneTexture;
uniform float saturation;
uniform vec3 tint;
uniform float contrast;
uniform float brightness;

in vec2 vUv;
out vec4 fragColor;

// Convert RGB to HSV
vec3 rgb2hsv(vec3 c) {
    vec4 K = vec4(0.0, -1.0 / 3.0, 2.0 / 3.0, -1.0);
    vec4 p = mix(vec4(c.bg, K.wz), vec4(c.gb, K.xy), step(c.b, c.g));
    vec4 q = mix(vec4(p.xyw, c.r), vec4(c.r, p.yzx), step(p.x, c.r));

    float d = q.x - min(q.w, q.y);
    float e = 1.0e-10;
    return vec3(abs(q.z + (q.w - q.y) / (6.0 * d + e)), d / (q.x + e), q.x);
}

// Convert HSV to RGB
vec3 hsv2rgb(vec3 c) {
    vec4 K = vec4(1.0, 2.0 / 3.0, 1.0 / 3.0, 3.0);
    vec3 p = abs(fract(c.xxx + K.xyz) * 6.0 - K.www);
    return c.z * mix(K.xxx, clamp(p - K.xxx, 0.0, 1.0), c.y);
}

void main() {
    vec4 sceneColor = texture(sceneTexture, vUv);

    // Brightness and contrast
    vec3 color = (sceneColor.rgb - 0.5) * contrast + 0.5 + brightness;

    // Saturation adjustment
    vec3 hsv = rgb2hsv(color);
    hsv.y *= saturation;
    color = hsv2rgb(hsv);

    // Color tint
    color *= tint;

    // Vignette
    vec2 center = vUv - 0.5;
    float vignette = 1.0 - dot(center, center) * 0.8;
    color *= vignette;

    fragColor = vec4(color, 1.0);
}


// ============================================================================
// DEPTH OF FIELD (BOKEH)
// ============================================================================

// Fragment shader - Hexagonal bokeh depth of field
#version 300 es
precision highp float;

uniform sampler2D sceneTexture;
uniform sampler2D depthTexture;
uniform float focusDistance;
uniform float focusRange;
uniform float bokehRadius;

in vec2 vUv;
out vec4 fragColor;

const int RING_COUNT = 5;
const int SAMPLES_PER_RING = 6;

// Hexagonal blur pattern
vec2 hexPattern(int i, int j) {
    float angle = float(j) / float(SAMPLES_PER_RING) * 6.28318530718;
    float radius = float(i) / float(RING_COUNT);
    return vec2(cos(angle), sin(angle)) * radius;
}

void main() {
    float depth = texture(depthTexture, vUv).r * 100.0;  // Linear depth
    vec2 texelSize = 1.0 / vec2(textureSize(sceneTexture, 0));

    // Calculate circle of confusion
    float coc = abs(depth - focusDistance) / focusRange;
    coc = clamp(coc, 0.0, 1.0);

    vec4 color = vec4(0.0);
    float totalWeight = 0.0;

    // Hexagonal blur
    for (int ring = 0; ring < RING_COUNT; ring++) {
        for (int sample = 0; sample < SAMPLES_PER_RING; sample++) {
            vec2 offset = hexPattern(ring, sample) * coc * bokehRadius * texelSize;
            vec4 sampleColor = texture(sceneTexture, vUv + offset);

            float weight = 1.0;  // Could add brightness-based weighting for bokeh highlights
            color += sampleColor * weight;
            totalWeight += weight;
        }
    }

    // Center sample (always included)
    vec4 centerColor = texture(sceneTexture, vUv);
    color += centerColor * 2.0;
    totalWeight += 2.0;

    color /= totalWeight;

    fragColor = color;
}


// ============================================================================
// BLOOM SHADER
// ============================================================================

// Fragment shader - Bright pass extraction
#version 300 es
precision highp float;

uniform sampler2D sceneTexture;
uniform float threshold;
uniform float intensity;

in vec2 vUv;
out vec4 fragColor;

void main() {
    vec4 color = texture(sceneTexture, vUv);

    // Extract bright pixels
    float brightness = dot(color.rgb, vec3(0.2126, 0.7152, 0.0722));
    float bloom = smoothstep(threshold, threshold + 0.5, brightness);

    vec3 bloomColor = color.rgb * bloom * intensity;

    fragColor = vec4(bloomColor, 1.0);
}

// Fragment shader - Gaussian blur (separable)
#version 300 es
precision highp float;

uniform sampler2D bloomTexture;
uniform vec2 direction;  // (1,0) for horizontal, (0,1) for vertical

in vec2 vUv;
out vec4 fragColor;

const float weights[5] = float[](0.227027, 0.1945946, 0.1216216, 0.054054, 0.016216);

void main() {
    vec2 texelSize = 1.0 / vec2(textureSize(bloomTexture, 0));
    vec3 result = texture(bloomTexture, vUv).rgb * weights[0];

    for (int i = 1; i < 5; i++) {
        vec2 offset = direction * texelSize * float(i);
        result += texture(bloomTexture, vUv + offset).rgb * weights[i];
        result += texture(bloomTexture, vUv - offset).rgb * weights[i];
    }

    fragColor = vec4(result, 1.0);
}


// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// Linear to sRGB conversion
vec3 linearToSRGB(vec3 linear) {
    return pow(linear, vec3(1.0 / 2.2));
}

// sRGB to linear conversion
vec3 sRGBToLinear(vec3 srgb) {
    return pow(srgb, vec3(2.2));
}

// ACES tone mapping
vec3 acesToneMapping(vec3 color) {
    const float a = 2.51;
    const float b = 0.03;
    const float c = 2.43;
    const float d = 0.59;
    const float e = 0.14;

    return clamp((color * (a * color + b)) / (color * (c * color + d) + e), 0.0, 1.0);
}

// Reinhard tone mapping
vec3 reinhardToneMapping(vec3 color) {
    return color / (1.0 + color);
}

// Film grain
float filmGrain(vec2 uv, float time) {
    float noise = fract(sin(dot(uv, vec2(12.9898, 78.233)) + time) * 43758.5453);
    return noise * 2.0 - 1.0;
}
