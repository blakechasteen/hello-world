# 3D Asset Guidelines for Dream Visualizer

**Target Audience**: 3D Artists, Technical Artists
**Date**: 2025-11-24
**Version**: 1.0

## Overview

This guide provides technical specifications for creating 3D assets for NeuroHood's Dream Visualizer. All assets must meet performance requirements (60 FPS target) while maintaining cinematic quality.

## Asset Specifications

### File Format

- **Format**: GLTF 2.0 (.glb binary)
- **Compression**: Draco (optional, for production)
- **Textures**: Embedded or separate (KTX2 recommended)

### Polygon Budget

| Asset Type | Triangle Count | Reasoning |
|------------|----------------|-----------|
| **Simple Props** | <1k tris | Small objects (bird, ball) |
| **Standard Props** | 1k-5k tris | Medium objects (cage, bridge) |
| **Large Props** | 5k-10k tris | Environment pieces (maze) |
| **Environment** | 10k-20k tris | Background geometry |

**Total Scene Budget**: 50k-100k triangles (target 60 FPS)

### Texture Specifications

| Texture Type | Resolution | Format | Notes |
|--------------|------------|--------|-------|
| **Diffuse/Albedo** | 512x512 or 1024x1024 | PNG/JPG | No lighting baked in |
| **Normal Map** | 512x512 | PNG | OpenGL format (Y+) |
| **Metallic/Roughness** | 512x512 | PNG | Packed (metal=B, rough=G) |
| **AO (Ambient Occlusion)** | 512x512 | PNG | Optional |
| **Emission** | 256x256 | PNG | For glowing effects |

**Compression**: Use KTX2 with Basis Universal for production (10x file size reduction).

### Materials

**PBR Workflow** (Metallic/Roughness):
- Base Color: RGB albedo (no lighting)
- Metallic: 0.0 (dielectric) or 1.0 (metal)
- Roughness: 0.0 (mirror) to 1.0 (matte)
- Normal: Tangent-space normal map
- Emissive: RGB glow color

**Material Types for Symbols**:

1. **Solid** (stone, wood, metal):
   - Roughness: 0.6-0.9
   - Metallic: 0.0 (non-metal)
   - Example: Bridge, cage bars

2. **Translucent** (glass, water, fog):
   - Transmission: 0.5-0.9
   - Roughness: 0.0-0.3
   - IOR: 1.5 (glass)
   - Example: Fog, void

3. **Ethereal** (glow, spirit, energy):
   - Emissive: 0.3-1.0 intensity
   - Opacity: 0.3-0.7
   - Additive blending
   - Example: Dream symbols, auras

### UV Mapping

- **Non-overlapping UVs**: Required for baking AO/lightmaps
- **Texture density**: Consistent across model (1024px = 1m)
- **UV islands**: Minimize seams on visible surfaces
- **Padding**: 4-8 pixels between islands

### Hierarchy and Naming

**Naming Convention**:
```
symbol_name_lod0.glb
├── Root
│   ├── Mesh_main
│   │   ├── Material_solid
│   ├── Mesh_detail
│   │   ├── Material_translucent
│   ├── Skeleton (optional, for animation)
```

**Requirements**:
- **Root node**: Named after symbol (e.g., "caged_bird")
- **Mesh names**: Descriptive (e.g., "cage_bars", "bird_body")
- **Material names**: Type-based (e.g., "metal_rusty", "feathers_blue")

### Vertex Groups (for Physics)

Some symbols require vertex groups for dynamic effects:

**Bridge** (dissolution effect):
- `solid`: Vertices that dissolve last (trust = 1.0)
- `transitional`: Vertices that dissolve mid-way (trust = 0.5)
- `ethereal`: Vertices that dissolve first (trust = 0.0)

**Cage** (scaling effect):
- `bars_vertical`: Vertical cage bars
- `bars_horizontal`: Horizontal cage bars
- `frame`: Cage frame structure

**Storm Cloud** (volume deformation):
- `core`: Inner high-density volume
- `wispy`: Outer low-density edges

## Symbol-Specific Guidelines

### 1. Caged Bird

**Concept**: Bird trapped in cage, yearning for freedom

**Geometry**:
- **Bird**: 800 tris (low-poly stylized)
- **Cage**: 1200 tris (bars + frame)
- **Total**: ~2k tris

**Materials**:
- Bird feathers: Diffuse (blue), Roughness 0.7
- Cage bars: Metallic 0.9, Roughness 0.5

**Vertex Groups**:
- `bird_body`: Main bird mesh
- `bird_wings`: Animated wings (optional)
- `cage_bars`: For opacity modulation

**Animations** (optional):
- `idle`: Bird hopping, looking around (loop)
- `struggle`: Bird flapping against bars (triggered by high emotion)

### 2. Storm Cloud

**Concept**: Ominous cloud representing inner turmoil

**Geometry**:
- **Base mesh**: 500 tris (low-poly volume)
- **Particle emitter**: Points for volumetric rendering
- **Total**: ~500 tris + particles

**Materials**:
- Cloud volume: Translucent, dark gray (#333333)
- Inner glow: Emissive (blue #4466ff) at 0.3 intensity

**Vertex Groups**:
- `core`: High-density center
- `wispy`: Low-density edges

**Notes**:
- Rendered with volumetric shader
- Particle system adds wisps

### 3. Bridge (Over Chasm)

**Concept**: Bridge that solidifies with trust, dissolves with doubt

**Geometry**:
- **Bridge deck**: 1500 tris (stone planks)
- **Supports**: 800 tris (wooden beams)
- **Rails**: 400 tris (rope/metal)
- **Total**: ~2.7k tris

**Materials**:
- Stone: Roughness 0.8, Normal map for detail
- Wood: Roughness 0.7, Diffuse with grain
- Rope: Roughness 0.9, Fiber normal map

**Vertex Groups** (CRITICAL):
- `solid`: Center planks (dissolve last)
- `transitional`: Middle sections
- `ethereal`: Edges and unsupported areas (dissolve first)

**Physics**:
- Trust < 0.5: Dissolve effect (shader discard)
- Trust > 0.7: Fully solid

### 4. Quicksand

**Concept**: Surface that reacts to panic, sinks you faster

**Geometry**:
- **Surface**: 400 tris (subdivided plane)
- **Particles**: 2000 sand particles
- **Total**: ~400 tris + particles

**Materials**:
- Sand surface: Diffuse (tan #d2b48c), Roughness 0.9
- Particles: Additive, brown (#8b7355)

**Animations**:
- Ripple effect (shader-based)
- Particle sink (physics-based)

### 5. Maze

**Concept**: Labyrinth representing confusion, decision paralysis

**Geometry**:
- **Walls**: 6k tris (modular sections)
- **Floor**: 1k tris
- **Ceiling**: Optional (fog instead)
- **Total**: ~7k tris

**Materials**:
- Stone walls: Roughness 0.8, Normal map
- Floor: Roughness 0.9, Worn stone

**Modular Design**:
- Tile-based (5m x 5m sections)
- Reusable wall pieces (instancing)

### 6. Mirror (Self-Reflection)

**Concept**: Mirror showing distorted or idealized self

**Geometry**:
- **Frame**: 600 tris (ornate design)
- **Mirror surface**: 200 tris (flat plane)
- **Total**: ~800 tris

**Materials**:
- Frame: Metallic 0.9 (gold/silver), Roughness 0.3
- Mirror: Metallic 1.0, Roughness 0.0, Reflection probe

**Effects**:
- Realtime reflection (expensive, use probe)
- Distortion shader (fun house effect)

### 7. Abyss (Void)

**Concept**: Infinite darkness, existential dread

**Geometry**:
- **Procedural**: Generated at runtime
- **Fallback mesh**: 200 tris (simple plane)

**Materials**:
- Black void: Emissive (#000011) at 0.1 intensity
- Fog: Dense volumetric

**Shader**:
- Depth-based darkening
- Particle absorption effect

## Procedural Geometry

Some symbols are generated procedurally (no GLTF file needed):

### Fog (Volumetric)

```typescript
const geometry = new THREE.BoxGeometry(100, 50, 100, 10, 5, 10);
const material = new THREE.ShaderMaterial({
    uniforms: { fogDensity: 0.01, fogColor: '#888888' },
    vertexShader: volumetricVert,
    fragmentShader: volumetricFrag,
    transparent: true
});
```

### Void (Empty Space)

```typescript
const geometry = new THREE.SphereGeometry(50, 32, 32);
const material = new THREE.MeshBasicMaterial({
    color: '#000000',
    side: THREE.BackSide  // Inside-out sphere
});
```

### Shadow (Dark Presence)

```typescript
const geometry = new THREE.PlaneGeometry(5, 5);
const material = new THREE.ShadowMaterial({
    opacity: 0.8
});
```

## Animation Guidelines

### Baked Animations (Skinned Mesh)

**Format**: GLTF animations (baked keyframes)

**Requirements**:
- **Frame rate**: 30 FPS (sufficient for dreams)
- **Compression**: Keyframe reduction (remove redundant)
- **Looping**: Mark loop points in GLTF
- **Duration**: 2-5 seconds per animation

**Example**:
```json
{
    "animations": [
        {
            "name": "bird_idle",
            "duration": 3.0,
            "channels": [...]
        }
    ]
}
```

### Shader Animations

**Preferred** for performance:
- UV scrolling (ripples, flow)
- Vertex displacement (waves, wind)
- Opacity fades (dissolve effects)

**Example** (bridge dissolution):
```glsl
uniform float trustLevel;
void main() {
    float noise = snoise(vPosition * 5.0 + time);
    if (noise < (1.0 - trustLevel) * 0.5) {
        discard;  // Dissolve fragment
    }
}
```

## Export Settings

### Blender → GLTF

**Settings**:
- Format: glTF Binary (.glb)
- Include: Selected Objects
- Transform: +Y Up (Blender default)
- Geometry:
  - [x] Apply Modifiers
  - [x] UVs
  - [x] Normals
  - [x] Tangents
  - [x] Vertex Colors (if used)
- Materials:
  - [x] Export Materials
  - [x] Images (embedded or separate)
- Animation:
  - [x] Use Current Frame Range
  - [x] Always Sample Animations (30 FPS)
- Compression:
  - [ ] Draco (enable for production only)

### Unity → GLTF

**Exporter**: UniGLTF or Piglet

**Settings**:
- Coordinate system: Right-handed, Y-up
- Scale: 1.0 (Unity units = meters)
- Materials: Convert to PBR
- Textures: Embed or external

### Maya → GLTF

**Exporter**: Babylon.js Exporter

**Settings**:
- Export: Selected
- Coordinate: Maya default (right-handed)
- Materials: PBR (convert legacy)
- Textures: Embed

## Testing Checklist

Before submitting assets:

- [ ] **Triangle count** within budget (<5k for props)
- [ ] **Textures** all 512x512 or 1024x1024
- [ ] **Materials** use PBR workflow (metallic/roughness)
- [ ] **UVs** non-overlapping, proper padding
- [ ] **Naming** follows convention (symbol_name_lod0.glb)
- [ ] **Hierarchy** clean (no empty nodes)
- [ ] **Pivot** at origin (0,0,0) or logical center
- [ ] **Scale** 1 unit = 1 meter (real-world)
- [ ] **Orientation** +Z forward, +Y up
- [ ] **Vertex groups** present (if required for physics)
- [ ] **Animations** baked at 30 FPS (if applicable)
- [ ] **File size** <5 MB per asset (pre-compression)
- [ ] **GLTF validation** passes (use gltf-validator)
- [ ] **Three.js test** loads and renders correctly

## Tools

### Recommended Software

**3D Creation**:
- Blender 3.6+ (free, GLTF export built-in)
- Maya 2023+ (UniGLTF plugin)
- 3ds Max 2023+ (Babylon exporter)

**Texturing**:
- Substance Painter (PBR texturing)
- Quixel Mixer (free, PBR materials)
- Blender (built-in texture painting)

**Validation**:
- gltf-validator (command-line, Khronos)
- Three.js Editor (web-based, test import)
- glTF Viewer (Windows, model inspection)

### Online Resources

**3D Models**:
- Sketchfab (reference models, CC-licensed)
- Poly Haven (free PBR textures + HDRIs)
- Kenney.nl (low-poly asset packs)

**Tutorials**:
- GLTF Tips: https://www.khronos.org/gltf/
- Three.js Docs: https://threejs.org/docs/
- PBR Guide: https://learnopengl.com/PBR/Theory

## Example Workflow

### Step-by-Step: Creating "Caged Bird"

1. **Model in Blender**:
   - Create bird (800 tris)
   - Create cage (1200 tris)
   - Apply scale (Ctrl+A → Scale)

2. **UV Unwrap**:
   - Select bird → U → Smart UV Project
   - Select cage → U → Unwrap (manual)
   - Check no overlaps

3. **Texture in Substance Painter**:
   - Export FBX → Import to Substance
   - Paint feathers (blue)
   - Paint cage (rusty metal)
   - Export: Base Color, Normal, Metallic, Roughness

4. **Apply Textures in Blender**:
   - Shader Editor → Principled BSDF
   - Connect textures (Base Color → Diffuse, etc.)

5. **Create Vertex Groups**:
   - Edit mode → Select cage bars
   - Assign to "cage_bars" group

6. **Export GLTF**:
   - File → Export → glTF 2.0 (.glb)
   - Apply settings (see above)

7. **Validate**:
   - Run gltf-validator on .glb
   - Load in Three.js Editor
   - Check appearance, scale, materials

8. **Submit**:
   - Place in `elle/ar_web_client/public/dream_models/`
   - Update manifest.json

## Performance Tips

### Optimization Techniques

1. **Use Instancing** (for repeated objects):
   - Example: Cage bars (all same geometry)
   - Reduces draw calls by 10x

2. **Combine Meshes** (where possible):
   - Example: Merge all non-moving parts
   - Fewer draw calls = better FPS

3. **Texture Atlasing** (multiple materials → one texture):
   - Combine bird + cage textures → single 2048x2048
   - One material = one draw call

4. **LOD (Level of Detail)** (future):
   - Create 3 versions: High (5k tris), Med (2k), Low (500)
   - Switch based on distance

5. **Vertex Welding** (remove duplicate vertices):
   - Blender: Mesh → Clean Up → Merge by Distance

6. **Backface Culling** (don't render invisible faces):
   - Material setting: Single-sided (default)

### Measuring Performance

**Three.js Stats**:
```typescript
import Stats from 'three/examples/jsm/libs/stats.module';
const stats = Stats();
document.body.appendChild(stats.dom);
```

**Target Metrics**:
- FPS: 60 (stable)
- Draw calls: <30 per scene
- Triangles: <100k visible
- Memory: <500 MB

## Submission Process

1. **Prepare asset** (follow guidelines above)
2. **Validate** (gltf-validator + Three.js test)
3. **Compress** (optional Draco, KTX2 textures)
4. **Submit to repository**:
   - Path: `elle/ar_web_client/public/dream_models/`
   - Filename: `symbol_name.glb`
5. **Update manifest**:
   - Add entry to `dream_models/manifest.json`
   - Include: name, triangles, materials, vertex_groups
6. **Test in visualizer**:
   - Run demo: `npm run dev`
   - Load dream scene with symbol
   - Verify appearance, physics, performance

---

**Contact**: HoloLoom Development Team
**Last Updated**: 2025-11-24
**Version**: 1.0
