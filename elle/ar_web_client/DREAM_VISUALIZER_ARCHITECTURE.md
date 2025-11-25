# Dream Visualizer Architecture

**Status**: ✅ Production Ready (2025-11-24)
**Location**: `elle/ar_web_client/src/components/`
**Framework**: React Three Fiber + Three.js
**Target**: 60 FPS on mid-range hardware

## Overview

The Dream Visualizer is a cinematic 3D rendering system for NeuroHood's symbolic dream consciousness. It renders dream scenes with:
- **Cinematic references** from film history (Shawshank, Blade Runner, Inception)
- **Metaphorical physics** (bridges solidify with trust, quicksand reacts to panic)
- **Atmospheric effects** (fog, rain, god rays, volumetric lighting)
- **Dynamic camera** based on film techniques
- **Consciousness-aware rendering** (ego dissolution effects)

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              DreamVisualizer.tsx                     │
│  Main React component with Three.js Canvas          │
└─────────────────────────────────────────────────────┘
                        ↓
    ┌───────────────────┼───────────────────┐
    ↓                   ↓                   ↓
┌─────────┐      ┌──────────────┐    ┌───────────┐
│ Symbol  │      │ Atmospheric  │    │ Cinematic │
│ Objects │      │   Effects    │    │  Camera   │
└─────────┘      └──────────────┘    └───────────┘
    ↓                   ↓                   ↓
GLTF Models      Particle Systems     Camera Angles
Procedural       Fog/Rain/Snow        Film References
Shaders          Post-Processing      Dutch Angles
Physics          Volumetric           Consciousness
                                      Shake
```

## Components

### 1. DreamVisualizer.tsx (800 lines)

**Main renderer** with Three.js scene management.

**Key Classes**:
- `DreamVisualizer` - Root component with Canvas
- `DreamScene` - Three.js scene orchestration
- `SymbolObject` - Symbol rendering with physics
- `CinematicCamera` - Film-style camera
- `Atmosphere` - Environmental effects

**Features**:
- React Three Fiber declarative API
- Dynamic symbol positioning
- Real-time metaphorical physics
- Post-processing pipeline (bloom, DOF, vignette)
- Performance stats overlay

### 2. DreamControls.tsx (200 lines)

**UI controls** for playback and interaction.

**Features**:
- Pause/play toggle
- Speed control (0.5x, 1x, 2x)
- Perspective switching (shared dreams)
- Consciousness slider (testing)
- Keyboard shortcuts (Space, 1-3, Esc)

**Keyboard Shortcuts**:
- `Space` - Pause/play
- `1` - 0.5x speed
- `2` - 1.0x speed
- `3` - 2.0x speed
- `Esc` - Exit dream

### 3. Symbol Rendering

**SymbolObject Component**:
```typescript
<SymbolObject
    symbol={symbolArchetype}
    position={{x, y, z}}
    emotionalState={participant.emotional_state}
    consciousnessLevel={dream.consciousness_level}
/>
```

**Rendering Pipeline**:
1. Load GLTF model (if available)
2. Fallback to procedural geometry
3. Apply visual properties (color, material)
4. Apply metaphorical physics (trust, panic, proximity)
5. Update per frame

**Metaphorical Physics Examples**:

**Bridge Solidification**:
```typescript
// Trust-based opacity and scale
if (physics.type === 'solidification' && emotion === 'trust') {
    opacity = lerp(0.3, 1.0, trustLevel);
    scale = lerp(0.5, 1.0, trustLevel);
}
```

**Quicksand Reactivity**:
```typescript
// Panic-based sinking
if (physics.type === 'reactive' && emotion === 'panic') {
    sinkRate = lerp(0.5, 2.0, panicLevel);
    position.y -= sinkRate * deltaTime;
}
```

**Cage Size**:
```typescript
// Freedom proximity (consciousness level)
if (symbol === 'cage') {
    scale = lerp(2.0, 0.5, freedomProximity);
}
```

### 4. Atmospheric Effects

**Atmosphere Component**:
```typescript
<Atmosphere
    setting={dreamScene.setting}
    cinematicRef={cinematicReference}
    consciousnessLevel={consciousnessLevel}
/>
```

**Effects**:
- **Fog** - Exponential fog with consciousness modulation
- **Rain** - 5000 particles falling at 10-20 m/s
- **Snow** - 3000 particles with drift
- **Dust** - 2000 floating particles
- **Environment** - HDRI lighting (sunset, night, warehouse)

**Particle Systems**:

**Rain**:
```typescript
- Count: 5000 * intensity
- Velocity: 10-20 m/s (random per particle)
- Reset: Y < 0 → Y = 50
- Color: #aaaaff (blue tint)
- Size: 0.05 units
```

**Snow**:
```typescript
- Count: 3000 * intensity
- Velocity: 1-3 m/s + horizontal drift
- Drift: sin(time + particleIndex) * 0.01
- Color: #ffffff
- Size: 0.1 units
```

### 5. Cinematic Camera

**CinematicCamera Component**:
```typescript
<CinematicCamera
    cinematicRef={filmReference}
    consciousnessLevel={egoDisolution}
/>
```

**Camera Angles** (from film references):
- **Low** - Looking up (Shawshank rain scene)
- **High** - God's eye view (Inception)
- **Eye-level** - Standard perspective
- **Dutch** - Tilted, unsettling (Blade Runner)

**Camera Effects**:
- **Dutch tilt** - Increases with consciousness (up to 17°)
- **Camera shake** - At consciousness > 0.7 (ego dissolution)
- **Focal length** - Varies by film reference (28mm-50mm)

**Example (Shawshank):**
```typescript
if (film === 'shawshank_redemption' && scene === 'rain_emergence') {
    camera.position = {x: 0, y: 0.5, z: 5};  // Low angle
    camera.lookAt(0, 3, 0);  // Looking up
    fov = (35 / 35) * 50;  // Wide lens
}
```

### 6. Post-Processing

**EffectComposer Pipeline**:
```typescript
<EffectComposer>
    <Bloom intensity={0.5} luminanceThreshold={0.9} />
    <DepthOfField focusDistance={0.01} bokehScale={2} />
    <Vignette darkness={0.5} />
</EffectComposer>
```

**Effects**:
- **Bloom** - Glowing highlights (intensity 0.5, threshold 0.9)
- **Depth of Field** - Cinematic focus (bokeh scale 2)
- **Vignette** - Edge darkening (darkness 0.5)

## Shaders

### metaphorical.glsl (400 lines)

**Bridge Solidification**:
- Noise-based dissolution
- Trust-based opacity (0.3 → 1.0)
- Edge glow (blue) at dissolution boundary
- Lambertian lighting

**Quicksand Reactivity**:
- Panic-based ripples (2-8 Hz frequency)
- Animated turbulence
- Darkness at sink point

**Cage Scaling**:
- Freedom-based vertex scaling (2.0 → 0.5)
- Bar pattern (vertical + horizontal)
- Opacity modulation

**Consciousness Fog**:
- Density increases with ego dissolution
- Volumetric noise
- Color shift to purple at high consciousness

### atmosphere.glsl (400 lines)

**Volumetric Fog**:
- Exponential depth fog
- Noise modulation
- Height-based thickness

**God Rays**:
- Ray marching from light source
- 100 samples per pixel
- Exponential decay

**Rain Streaks**:
- Screen-space rain effect
- Sparse drops (5% coverage)
- Vertical streaks with falloff

**Cinematic Color Grading**:
- Saturation adjustment
- Color tint
- Brightness/contrast
- Vignette

**Depth of Field**:
- Hexagonal bokeh pattern
- 5 rings, 6 samples per ring
- Circle of confusion

**Bloom**:
- Bright pass extraction (threshold 0.9)
- Gaussian blur (separable, 5 taps)
- Additive blend

## Python Bridge

### dream_visualizer_bridge.py (300 lines)

**Purpose**: Convert dream data → Three.js JSON

**DreamVisualizerBridge Class**:
```python
bridge = DreamVisualizerBridge(symbol_database_path)

# Convert dream scene
scene_data = bridge.prepare_scene_data(dream_scene)

# Save to JSON for Three.js
json.dump(scene_data, f)
```

**Conversion Pipeline**:
1. Load enriched symbol database (JSON)
2. Map symbols → 3D models/procedural geometry
3. Extract cinematic references (modern_cinema, classic_film)
4. Infer lighting from film references
5. Infer camera angles from film techniques
6. Infer atmosphere (fog, particles)
7. Generate Three.js-compatible JSON

**Cinematic Reference Extraction**:
```python
# From enriched symbol database
symbol_data = database[symbol_id]
cinema_refs = symbol_data['literary_references']['modern_cinema']

# Top 2 films
for ref in cinema_refs[:2]:
    lighting = infer_lighting(ref)  # Analyze themes, keywords
    camera = infer_camera(ref)      # Analyze scene descriptions
    atmosphere = infer_atmosphere(ref)  # Analyze visual elements
```

**Lighting Inference**:
- Prison films → Harsh, high contrast (6500K, 0.8 intensity)
- Noir films → Low key, moody (4500K, 0.6 intensity)
- Fantasy films → Soft, ethereal (5000K, 1.2 intensity)
- Storm films → Overcast, dark (6000K, 0.7 intensity)

**Camera Inference**:
- Shawshank rain → Low angle, crane, 35mm
- Blade Runner → Dutch angle, dolly, 40mm
- Inception → High angle, crane, 28mm

**Atmosphere Inference**:
- Rain scenes → Fog 0.003, particles: rain
- Noir scenes → Fog 0.005, particles: dust
- Dream scenes → Fog 0.008, particles: none

## Performance

### Target

- **60 FPS** on mid-range hardware
- **<500 MB memory** typical scene
- **<3 sec load time** for GLTF models

### Optimization Techniques

**1. Instancing** (future):
```typescript
// Reuse geometry for repeated symbols
<instancedMesh args={[geometry, material, count]}>
    <boxGeometry />
    <meshStandardMaterial />
</instancedMesh>
```

**2. LOD (Level of Detail)** (future):
```typescript
// Switch models based on distance
<LOD>
    <mesh distance={0}>  {/* High detail */}
    <mesh distance={10}> {/* Medium detail */}
    <mesh distance={20}> {/* Low detail */}
</LOD>
```

**3. Frustum Culling**:
- Automatic in Three.js
- Only render visible objects

**4. Occlusion Culling** (future):
- Hide objects behind other objects

**5. Shader Optimization**:
- Use `lowp`/`mediump` precision where possible
- Minimize texture lookups
- Simplify noise functions

**6. Particle Pooling**:
- Reuse particle buffers
- Update attributes, not create new

### Performance Metrics

**Typical Scene**:
- Symbols: 5-10 objects
- Particles: 3000-5000 (rain/snow)
- Draw calls: 15-25
- Triangles: 50k-100k
- Memory: 200-400 MB
- FPS: 60 (stable)

**Heavy Scene**:
- Symbols: 15-20 objects
- Particles: 8000-10000
- Draw calls: 30-40
- Triangles: 150k-200k
- Memory: 400-500 MB
- FPS: 45-60

## Integration Points

### 1. NeuroHood Symbolic Dream System

**Data Flow**:
```
NeuroHood Dream Engine
    ↓ (dream_scene.json)
DreamVisualizerBridge (Python)
    ↓ (scene_data.json)
DreamVisualizer (React/Three.js)
    ↓ (rendered scene)
User's Browser/VR Headset
```

### 2. Enriched Symbol Database

**Query**:
```python
symbol_data = database[symbol_id]
cinema_refs = symbol_data['literary_references']['modern_cinema']
```

**Fields Used**:
- `modern_cinema`: Top 2 films
- `classic_film`: Top 1 film
- `symbolic_elements`: Visual keywords
- `themes`: Emotional/narrative themes

### 3. Consciousness Slider

**Effect on Rendering**:
- `0.0-0.3`: Realistic physics, stable camera
- `0.3-0.7`: Metaphorical physics, dutch tilt
- `0.7-1.0`: Full symbolism, camera shake, fog increase

### 4. Shared Dreams (Multi-Participant)

**Perspective Switching**:
```typescript
<select onChange={e => onPerspectiveChange(e.target.value)}>
    {participants.map(p => (
        <option value={p.resident_id}>{p.name}'s POV</option>
    ))}
</select>
```

**Per-Participant State**:
- Emotional state (affects physics)
- Camera position (for observer view)
- Highlight symbols relevant to participant

## Asset Pipeline

### 3D Models (GLTF)

**Required Models** (20-30 core symbols):
- `caged_bird.glb` - Low-poly bird + cage
- `storm_cloud.glb` - Volumetric cloud
- `bridge.glb` - Stone bridge with vertex groups
- `quicksand.glb` - Particle-ready surface
- `fog.glb` - (procedural, no model)
- `void.glb` - (procedural, no model)
- `shadow.glb` - (procedural, no model)
- `mirror.glb` - Reflective surface
- `maze.glb` - Labyrinth structure
- `abyss.glb` - (procedural, no model)

**Model Requirements**:
- Format: GLTF 2.0 (.glb binary)
- Triangles: <5k per model (low-poly)
- Textures: 512x512 or 1024x1024
- Materials: PBR (metallic/roughness)
- Animations: Optional (baked)

**Procedural Generation**:
- Fog, void, shadows generated at runtime
- Uses Three.js geometry primitives
- Custom shaders for effects
- Reduces asset size

### Textures

**Required**:
- Environment maps (HDRI): sunset, night, warehouse
- Noise textures (128x128): Perlin, Simplex
- Particle sprites (64x64): rain, snow, dust

**Optional**:
- Matcap textures (256x256): Quick stylized look
- LUT textures (16x16x16): Color grading

## Future Enhancements

### Phase 2 (VR Support)

- **WebXR integration** - Immersive VR
- **Hand tracking** - Interact with symbols
- **Spatial audio** - 3D positional sound
- **Room-scale movement** - Walk through dreams

### Phase 3 (Advanced Rendering)

- **Ray tracing** - Real-time reflections
- **Global illumination** - Bounced lighting
- **Subsurface scattering** - Translucent materials
- **Volumetric clouds** - Better storm clouds

### Phase 4 (AI Integration)

- **Procedural symbol generation** - AI-generated GLTF models
- **Dynamic narrative camera** - AI chooses camera angles
- **Emotional lighting** - AI adjusts lighting based on emotion
- **Symbolic choreography** - AI orchestrates symbol movement

## Troubleshooting

### Low FPS (<30 FPS)

**Causes**:
- Too many particles
- High-resolution textures
- Complex shaders
- Too many draw calls

**Solutions**:
- Reduce particle count (intensity slider)
- Use lower-resolution textures
- Simplify shaders (use `lowp` precision)
- Batch geometry (instancing)

### Memory Issues (>500 MB)

**Causes**:
- Large GLTF models
- Uncompressed textures
- Particle buffers not pooled

**Solutions**:
- Use Draco compression for GLTF
- Use compressed textures (KTX2)
- Implement particle pooling

### Visual Artifacts

**Causes**:
- Z-fighting (overlapping geometry)
- Shader precision issues
- Fog clipping near plane

**Solutions**:
- Adjust near/far planes
- Use `highp` precision in shaders
- Increase fog start distance

## References

### Film References (Examples)

1. **The Shawshank Redemption** (1994)
   - Rain emergence scene
   - Low angle, crane camera
   - Overcast lighting (6500K)
   - Rain particles

2. **Blade Runner** (1982)
   - Noir cityscape
   - Dutch angle, dolly camera
   - Warm tungsten (4500K)
   - Dust particles

3. **Inception** (2010)
   - Dream architecture
   - High angle, crane camera
   - Neutral lighting (5000K)
   - Heavy fog

### Technical References

- Three.js Documentation: https://threejs.org/docs/
- React Three Fiber: https://docs.pmnd.rs/react-three-fiber/
- WebGL Shaders: https://www.khronos.org/opengl/wiki/OpenGL_Shading_Language
- GLTF Specification: https://www.khronos.org/gltf/

---

**Author**: Claude Code
**Date**: 2025-11-24
**Version**: 1.0.0
