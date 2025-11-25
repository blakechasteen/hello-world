/**
 * Metaphorical Physics Shaders - Custom GLSL shaders for dream symbolism
 *
 * Implements metaphorical physics effects:
 * - Bridge solidification (trust-based opacity/dissolution)
 * - Quicksand reactivity (panic-based sinking)
 * - Cage size modulation (freedom proximity)
 * - Fog density (consciousness-based thickness)
 *
 * All effects driven by emotional state and consciousness level.
 */

// ============================================================================
// BRIDGE SOLIDIFICATION SHADER
// ============================================================================

// Vertex shader
#version 300 es
in vec3 position;
in vec3 normal;
in vec2 uv;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat3 normalMatrix;

out vec3 vNormal;
out vec3 vPosition;
out vec2 vUv;

void main() {
    vNormal = normalize(normalMatrix * normal);
    vPosition = position;
    vUv = uv;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
}

// Fragment shader - Bridge solidification
#version 300 es
precision highp float;

uniform float trustLevel;  // 0.0-1.0
uniform float time;
uniform vec3 baseColor;
uniform vec3 fogColor;

in vec3 vNormal;
in vec3 vPosition;
in vec2 vUv;

out vec4 fragColor;

// 3D Perlin noise (simplified for performance)
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
    // Dissolve effect based on trust
    float dissolveFactor = mix(0.3, 1.0, trustLevel);

    // Animated noise pattern
    float noiseValue = noise(vPosition * 5.0 + time * 0.2);

    // Discard fragments below dissolution threshold
    if (noiseValue < (1.0 - trustLevel) * 0.5) {
        discard;
    }

    // Edge glow (reveal dissolving edges)
    float edgeFactor = smoothstep((1.0 - trustLevel) * 0.5 - 0.1, (1.0 - trustLevel) * 0.5, noiseValue);
    vec3 edgeColor = vec3(0.3, 0.6, 1.0);  // Blue glow

    // Blend base color with edge glow
    vec3 finalColor = mix(baseColor, edgeColor, (1.0 - edgeFactor) * 0.5);

    // Lighting (simple Lambertian)
    vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
    float diffuse = max(dot(vNormal, lightDir), 0.0);
    finalColor *= (0.3 + 0.7 * diffuse);

    // Fog
    float depth = gl_FragCoord.z / gl_FragCoord.w;
    float fogFactor = exp(-0.002 * depth);
    finalColor = mix(fogColor, finalColor, fogFactor);

    fragColor = vec4(finalColor, dissolveFactor);
}


// ============================================================================
// QUICKSAND REACTIVITY SHADER
// ============================================================================

// Fragment shader - Quicksand surface
#version 300 es
precision highp float;

uniform float panicLevel;  // 0.0-1.0
uniform float time;
uniform vec3 sandColor;

in vec3 vNormal;
in vec3 vPosition;
in vec2 vUv;

out vec4 fragColor;

float noise(vec3 p) {
    // Same noise function as above
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
    // Animated ripples based on panic
    float rippleSpeed = mix(0.5, 2.0, panicLevel);
    float rippleFreq = mix(2.0, 8.0, panicLevel);

    vec2 center = vec2(0.5, 0.5);
    float dist = distance(vUv, center);
    float ripple = sin(dist * rippleFreq - time * rippleSpeed) * 0.5 + 0.5;

    // Turbulence (higher when panicking)
    float turbulence = noise(vec3(vPosition.xz * 3.0, time * 0.5)) * panicLevel;

    // Darker at center (sinking point)
    float darkness = smoothstep(0.5, 0.0, dist) * panicLevel * 0.5;

    // Final color
    vec3 finalColor = sandColor * (0.7 + ripple * 0.3 - darkness);
    finalColor += turbulence * 0.2;

    fragColor = vec4(finalColor, 1.0);
}


// ============================================================================
// CAGE SCALE MODULATION (Vertex Shader)
// ============================================================================

// Vertex shader - Cage size based on freedom proximity
#version 300 es
in vec3 position;
in vec3 normal;

uniform mat4 modelViewMatrix;
uniform mat4 projectionMatrix;
uniform mat3 normalMatrix;
uniform float freedomProximity;  // 0.0-1.0 (0=trapped, 1=free)

out vec3 vNormal;
out vec3 vPosition;

void main() {
    vNormal = normalize(normalMatrix * normal);

    // Scale cage based on freedom (larger when trapped, smaller when free)
    float scale = mix(2.0, 0.5, freedomProximity);
    vec3 scaledPosition = position * scale;

    vPosition = scaledPosition;
    gl_Position = projectionMatrix * modelViewMatrix * vec4(scaledPosition, 1.0);
}

// Fragment shader - Cage bars with freedom-based transparency
#version 300 es
precision highp float;

uniform float freedomProximity;
uniform vec3 cageColor;

in vec3 vNormal;
in vec3 vPosition;

out vec4 fragColor;

void main() {
    // Bars opacity (more opaque when trapped)
    float opacity = mix(1.0, 0.3, freedomProximity);

    // Bar pattern (vertical and horizontal)
    float barWidth = 0.1;
    float barSpacing = 1.0;

    float verticalBar = step(mod(vPosition.x, barSpacing), barWidth);
    float horizontalBar = step(mod(vPosition.z, barSpacing), barWidth);
    float barMask = max(verticalBar, horizontalBar);

    // Discard non-bar fragments
    if (barMask < 0.5) {
        discard;
    }

    // Lighting
    vec3 lightDir = normalize(vec3(1.0, 1.0, 1.0));
    float diffuse = max(dot(vNormal, lightDir), 0.0);
    vec3 finalColor = cageColor * (0.3 + 0.7 * diffuse);

    fragColor = vec4(finalColor, opacity);
}


// ============================================================================
// CONSCIOUSNESS FOG SHADER
// ============================================================================

// Fragment shader - Ego dissolution fog
#version 300 es
precision highp float;

uniform float consciousnessLevel;  // 0.0-1.0
uniform float time;
uniform vec3 fogColor;
uniform vec3 sceneColor;

in vec3 vPosition;

out vec4 fragColor;

float noise(vec3 p) {
    // Same noise function
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
    // Fog density increases with consciousness level
    float fogDensity = mix(0.002, 0.01, consciousnessLevel);

    // Volumetric fog with noise
    float depth = gl_FragCoord.z / gl_FragCoord.w;
    float noiseValue = noise(vPosition * 0.5 + vec3(time * 0.1));
    float fogFactor = exp(-fogDensity * depth * (0.5 + noiseValue * 0.5));

    // Color shift at high consciousness (ego dissolution)
    vec3 dissolutionColor = mix(fogColor, vec3(0.8, 0.6, 1.0), consciousnessLevel * 0.5);

    vec3 finalColor = mix(dissolutionColor, sceneColor, fogFactor);
    fragColor = vec4(finalColor, 1.0);
}


// ============================================================================
// UTILITY FUNCTIONS
// ============================================================================

// Better 3D Perlin noise (more expensive but smoother)
vec3 permute(vec3 x) {
    return mod(((x * 34.0) + 1.0) * x, 289.0);
}

float snoise(vec3 v) {
    const vec2 C = vec2(1.0 / 6.0, 1.0 / 3.0);
    const vec4 D = vec4(0.0, 0.5, 1.0, 2.0);

    // First corner
    vec3 i  = floor(v + dot(v, C.yyy));
    vec3 x0 = v - i + dot(i, C.xxx);

    // Other corners
    vec3 g = step(x0.yzx, x0.xyz);
    vec3 l = 1.0 - g;
    vec3 i1 = min(g.xyz, l.zxy);
    vec3 i2 = max(g.xyz, l.zxy);

    vec3 x1 = x0 - i1 + C.xxx;
    vec3 x2 = x0 - i2 + C.yyy;
    vec3 x3 = x0 - D.yyy;

    // Permutations
    i = mod(i, 289.0);
    vec4 p = permute(permute(permute(
                i.z + vec4(0.0, i1.z, i2.z, 1.0))
              + i.y + vec4(0.0, i1.y, i2.y, 1.0))
              + i.x + vec4(0.0, i1.x, i2.x, 1.0));

    // Gradients
    float n_ = 1.0 / 7.0;
    vec3 ns = n_ * D.wyz - D.xzx;

    vec4 j = p - 49.0 * floor(p * ns.z * ns.z);

    vec4 x_ = floor(j * ns.z);
    vec4 y_ = floor(j - 7.0 * x_);

    vec4 x = x_ * ns.x + ns.yyyy;
    vec4 y = y_ * ns.x + ns.yyyy;
    vec4 h = 1.0 - abs(x) - abs(y);

    vec4 b0 = vec4(x.xy, y.xy);
    vec4 b1 = vec4(x.zw, y.zw);

    vec4 s0 = floor(b0) * 2.0 + 1.0;
    vec4 s1 = floor(b1) * 2.0 + 1.0;
    vec4 sh = -step(h, vec4(0.0));

    vec4 a0 = b0.xzyw + s0.xzyw * sh.xxyy;
    vec4 a1 = b1.xzyw + s1.xzyw * sh.zzww;

    vec3 p0 = vec3(a0.xy, h.x);
    vec3 p1 = vec3(a0.zw, h.y);
    vec3 p2 = vec3(a1.xy, h.z);
    vec3 p3 = vec3(a1.zw, h.w);

    // Normalize gradients
    vec4 norm = inversesqrt(vec4(dot(p0, p0), dot(p1, p1), dot(p2, p2), dot(p3, p3)));
    p0 *= norm.x;
    p1 *= norm.y;
    p2 *= norm.z;
    p3 *= norm.w;

    // Mix final noise value
    vec4 m = max(0.6 - vec4(dot(x0, x0), dot(x1, x1), dot(x2, x2), dot(x3, x3)), 0.0);
    m = m * m;
    return 42.0 * dot(m * m, vec4(dot(p0, x0), dot(p1, x1), dot(p2, x2), dot(p3, x3)));
}
