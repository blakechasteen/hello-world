/**
 * Dream Visualizer - Three.js-based cinematic dream renderer for NeuroHood
 *
 * Renders symbolic dreams with:
 * - Cinematic references from enriched symbol database
 * - Metaphorical physics (bridges solidify, quicksand reacts to panic)
 * - Atmospheric effects (fog, particles, volumetric lighting)
 * - Dynamic camera based on film references
 *
 * Architecture:
 * - React Three Fiber (declarative Three.js)
 * - Custom shaders for metaphorical effects
 * - Post-processing for cinematic look
 * - Performance optimized (60 FPS target)
 */

import React, { useRef, useEffect, useState, useMemo } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { OrbitControls, PerspectiveCamera, useGLTF, Environment, Stats } from '@react-three/drei';
import * as THREE from 'three';
import { EffectComposer, Bloom, DepthOfField, Vignette } from '@react-three/postprocessing';

// ============================================================================
// TYPE DEFINITIONS
// ============================================================================

interface Vector3 {
    x: number;
    y: number;
    z: number;
}

interface SymbolArchetype {
    symbol_id: string;
    name: string;
    category: string;
    visual_properties: {
        base_form: string;  // 'organic' | 'geometric' | 'abstract'
        primary_color: string;
        secondary_color?: string;
        material_type: string;  // 'solid' | 'translucent' | 'ethereal'
        scale: number;
    };
    metaphorical_physics: MetaphoricalPhysics;
    cinematic_references?: CinematicReference[];
}

interface MetaphoricalPhysics {
    type: string;  // 'solidification' | 'dissolution' | 'reactive' | 'gravitational'
    triggers: {
        emotion?: string;
        proximity?: string;
        interaction?: string;
    };
    parameters: {
        base_rate?: number;
        max_rate?: number;
        threshold?: number;
    };
}

interface CinematicReference {
    film: string;
    scene: string;
    visual_elements: string[];
    lighting: {
        type: string;  // 'overcast' | 'dramatic' | 'soft' | 'harsh'
        color_temp: number;  // Kelvin
        intensity: number;
    };
    camera: {
        angle: string;  // 'low' | 'high' | 'eye-level' | 'dutch'
        movement: string;  // 'static' | 'dolly' | 'crane' | 'handheld'
        focal_length: number;  // mm
    };
    atmosphere: {
        fog_density: number;
        particles?: string;  // 'rain' | 'snow' | 'dust'
        time_of_day: string;
    };
}

interface DreamSetting {
    environment: string;  // 'prison' | 'forest' | 'ocean' | 'void'
    lighting: string;  // 'twilight' | 'stormy' | 'starry' | 'harsh_sunlight'
    weather?: string;
    atmosphere: string;  // 'oppressive' | 'serene' | 'chaotic' | 'isolating'
}

interface DreamScene {
    scene_id: string;
    title: string;
    symbols: SymbolArchetype[];
    setting: DreamSetting;
    participants: Participant[];
    duration_seconds: number;
    consciousness_level: number;  // 0.0-1.0 (ego dissolution)
}

interface Participant {
    resident_id: string;
    name: string;
    perspective: 'first_person' | 'observer';
    emotional_state: {
        primary: string;
        intensity: number;
    };
}

export interface DreamVisualizerProps {
    dreamScene: DreamScene;
    participants: Participant[];
    onDreamEnd: () => void;
    isPaused?: boolean;
    playbackSpeed?: number;  // 0.5x, 1x, 2x
    activePerspective?: string;  // resident_id
}

// ============================================================================
// SYMBOL RENDERERS
// ============================================================================

/**
 * Generic symbol renderer - loads GLTF models or generates procedurally
 */
const SymbolObject: React.FC<{
    symbol: SymbolArchetype;
    position: Vector3;
    emotionalState: any;
    consciousnessLevel: number;
}> = ({ symbol, position, emotionalState, consciousnessLevel }) => {
    const meshRef = useRef<THREE.Group>(null);
    const [opacity, setOpacity] = useState(1.0);
    const [scale, setScale] = useState(1.0);

    // Load GLTF model if available
    const modelPath = `/dream_models/${symbol.name.toLowerCase().replace(' ', '_')}.glb`;
    let model: any = null;
    try {
        model = useGLTF(modelPath);
    } catch (e) {
        // Fallback to procedural generation
        model = null;
    }

    // Apply metaphorical physics
    useFrame((state, delta) => {
        if (!meshRef.current) return;

        const physics = symbol.metaphorical_physics;

        // Bridge solidification (trust-based)
        if (physics.type === 'solidification' && physics.triggers.emotion === 'trust') {
            const trustLevel = emotionalState.primary === 'trust' ? emotionalState.intensity : 0.0;
            const targetOpacity = THREE.MathUtils.lerp(0.3, 1.0, trustLevel);
            const targetScale = THREE.MathUtils.lerp(0.5, 1.0, trustLevel);

            setOpacity(THREE.MathUtils.lerp(opacity, targetOpacity, delta * 2.0));
            setScale(THREE.MathUtils.lerp(scale, targetScale, delta * 2.0));
        }

        // Quicksand reactive physics (panic-based)
        if (physics.type === 'reactive' && physics.triggers.emotion === 'panic') {
            const panicLevel = emotionalState.primary === 'panic' ? emotionalState.intensity : 0.0;
            const sinkRate = THREE.MathUtils.lerp(
                physics.parameters.base_rate || 0.5,
                physics.parameters.max_rate || 2.0,
                panicLevel
            );

            // Sink player (would affect camera in full implementation)
            meshRef.current.position.y -= sinkRate * delta;
        }

        // Cage size based on freedom proximity (metaphorical)
        if (symbol.name === 'cage') {
            const freedomProximity = consciousnessLevel;  // Higher = closer to ego dissolution/freedom
            const targetScale = THREE.MathUtils.lerp(2.0, 0.5, freedomProximity);
            setScale(THREE.MathUtils.lerp(scale, targetScale, delta));
        }

        // Apply scale
        meshRef.current.scale.setScalar(scale * symbol.visual_properties.scale);
    });

    // Get material properties
    const material = useMemo(() => {
        const props = symbol.visual_properties;
        const baseColor = new THREE.Color(props.primary_color);

        switch (props.material_type) {
            case 'translucent':
                return <meshPhysicalMaterial
                    color={baseColor}
                    transparent
                    opacity={opacity}
                    transmission={0.5}
                    roughness={0.2}
                    metalness={0.1}
                />;
            case 'ethereal':
                return <meshStandardMaterial
                    color={baseColor}
                    transparent
                    opacity={opacity * 0.6}
                    emissive={baseColor}
                    emissiveIntensity={0.3}
                />;
            default:  // solid
                return <meshStandardMaterial
                    color={baseColor}
                    transparent={opacity < 1.0}
                    opacity={opacity}
                    roughness={0.7}
                    metalness={0.3}
                />;
        }
    }, [symbol, opacity]);

    // Render GLTF model or procedural geometry
    if (model && model.scene) {
        return (
            <group ref={meshRef} position={[position.x, position.y, position.z]}>
                <primitive object={model.scene.clone()} />
            </group>
        );
    }

    // Fallback: procedural geometry based on base_form
    const geometry = useMemo(() => {
        switch (symbol.visual_properties.base_form) {
            case 'organic':
                return <sphereGeometry args={[1, 32, 32]} />;
            case 'geometric':
                return <boxGeometry args={[1, 1, 1]} />;
            case 'abstract':
                return <torusKnotGeometry args={[0.5, 0.2, 128, 16]} />;
            default:
                return <sphereGeometry args={[1, 32, 32]} />;
        }
    }, [symbol.visual_properties.base_form]);

    return (
        <group ref={meshRef} position={[position.x, position.y, position.z]}>
            <mesh>
                {geometry}
                {material}
            </mesh>
        </group>
    );
};

// ============================================================================
// ATMOSPHERIC EFFECTS
// ============================================================================

/**
 * Dynamic atmosphere based on dream setting and cinematic references
 */
const Atmosphere: React.FC<{
    setting: DreamSetting;
    cinematicRef?: CinematicReference;
    consciousnessLevel: number;
}> = ({ setting, cinematicRef, consciousnessLevel }) => {
    const { scene } = useThree();

    useEffect(() => {
        // Fog configuration
        const fogDensity = cinematicRef?.atmosphere.fog_density || 0.002;
        const fogColor = setting.lighting === 'twilight' ? 0x8888bb :
                        setting.lighting === 'stormy' ? 0x555555 :
                        setting.lighting === 'starry' ? 0x111133 :
                        0x888888;

        scene.fog = new THREE.FogExp2(fogColor, fogDensity * (1 + consciousnessLevel * 0.5));

        // Background color
        scene.background = new THREE.Color(fogColor);

        return () => {
            scene.fog = null;
        };
    }, [scene, setting, cinematicRef, consciousnessLevel]);

    return (
        <>
            {/* Particle effects */}
            {cinematicRef?.atmosphere.particles === 'rain' && <RainParticles intensity={0.8} />}
            {cinematicRef?.atmosphere.particles === 'snow' && <SnowParticles intensity={0.5} />}
            {cinematicRef?.atmosphere.particles === 'dust' && <DustParticles intensity={0.3} />}

            {/* Environment lighting */}
            <Environment preset={
                setting.lighting === 'twilight' ? 'sunset' :
                setting.lighting === 'stormy' ? 'night' :
                setting.lighting === 'starry' ? 'night' :
                'warehouse'
            } />
        </>
    );
};

/**
 * Rain particle system
 */
const RainParticles: React.FC<{ intensity: number }> = ({ intensity }) => {
    const particlesRef = useRef<THREE.Points>(null);
    const count = Math.floor(5000 * intensity);

    const particles = useMemo(() => {
        const positions = new Float32Array(count * 3);
        const velocities = new Float32Array(count);

        for (let i = 0; i < count; i++) {
            positions[i * 3] = (Math.random() - 0.5) * 50;
            positions[i * 3 + 1] = Math.random() * 50;
            positions[i * 3 + 2] = (Math.random() - 0.5) * 50;
            velocities[i] = Math.random() * 10 + 10;
        }

        return { positions, velocities };
    }, [count]);

    useFrame((state, delta) => {
        if (!particlesRef.current) return;

        const positions = particlesRef.current.geometry.attributes.position.array as Float32Array;

        for (let i = 0; i < count; i++) {
            positions[i * 3 + 1] -= particles.velocities[i] * delta;

            // Reset at bottom
            if (positions[i * 3 + 1] < 0) {
                positions[i * 3 + 1] = 50;
            }
        }

        particlesRef.current.geometry.attributes.position.needsUpdate = true;
    });

    return (
        <points ref={particlesRef}>
            <bufferGeometry>
                <bufferAttribute
                    attach="attributes-position"
                    count={count}
                    array={particles.positions}
                    itemSize={3}
                />
            </bufferGeometry>
            <pointsMaterial size={0.05} color={0xaaaaff} transparent opacity={0.6} />
        </points>
    );
};

/**
 * Snow particle system
 */
const SnowParticles: React.FC<{ intensity: number }> = ({ intensity }) => {
    const particlesRef = useRef<THREE.Points>(null);
    const count = Math.floor(3000 * intensity);

    const particles = useMemo(() => {
        const positions = new Float32Array(count * 3);
        const velocities = new Float32Array(count);

        for (let i = 0; i < count; i++) {
            positions[i * 3] = (Math.random() - 0.5) * 50;
            positions[i * 3 + 1] = Math.random() * 50;
            positions[i * 3 + 2] = (Math.random() - 0.5) * 50;
            velocities[i] = Math.random() * 2 + 1;
        }

        return { positions, velocities };
    }, [count]);

    useFrame((state, delta) => {
        if (!particlesRef.current) return;

        const positions = particlesRef.current.geometry.attributes.position.array as Float32Array;

        for (let i = 0; i < count; i++) {
            positions[i * 3 + 1] -= particles.velocities[i] * delta;
            positions[i * 3] += Math.sin(state.clock.elapsedTime + i) * 0.01;

            // Reset at bottom
            if (positions[i * 3 + 1] < 0) {
                positions[i * 3 + 1] = 50;
            }
        }

        particlesRef.current.geometry.attributes.position.needsUpdate = true;
    });

    return (
        <points ref={particlesRef}>
            <bufferGeometry>
                <bufferAttribute
                    attach="attributes-position"
                    count={count}
                    array={particles.positions}
                    itemSize={3}
                />
            </bufferGeometry>
            <pointsMaterial size={0.1} color={0xffffff} transparent opacity={0.8} />
        </points>
    );
};

/**
 * Dust particle system
 */
const DustParticles: React.FC<{ intensity: number }> = ({ intensity }) => {
    const particlesRef = useRef<THREE.Points>(null);
    const count = Math.floor(2000 * intensity);

    const particles = useMemo(() => {
        const positions = new Float32Array(count * 3);

        for (let i = 0; i < count; i++) {
            positions[i * 3] = (Math.random() - 0.5) * 50;
            positions[i * 3 + 1] = Math.random() * 30;
            positions[i * 3 + 2] = (Math.random() - 0.5) * 50;
        }

        return { positions };
    }, [count]);

    useFrame((state, delta) => {
        if (!particlesRef.current) return;

        const positions = particlesRef.current.geometry.attributes.position.array as Float32Array;

        for (let i = 0; i < count; i++) {
            positions[i * 3] += Math.sin(state.clock.elapsedTime * 0.5 + i) * 0.02 * delta;
            positions[i * 3 + 2] += Math.cos(state.clock.elapsedTime * 0.5 + i) * 0.02 * delta;
        }

        particlesRef.current.geometry.attributes.position.needsUpdate = true;
    });

    return (
        <points ref={particlesRef}>
            <bufferGeometry>
                <bufferAttribute
                    attach="attributes-position"
                    count={count}
                    array={particles.positions}
                    itemSize={3}
                />
            </bufferGeometry>
            <pointsMaterial size={0.05} color={0xccaa88} transparent opacity={0.3} />
        </points>
    );
};

// ============================================================================
// CINEMATIC CAMERA
// ============================================================================

/**
 * Dynamic camera based on cinematic references
 */
const CinematicCamera: React.FC<{
    cinematicRef?: CinematicReference;
    consciousnessLevel: number;
}> = ({ cinematicRef, consciousnessLevel }) => {
    const cameraRef = useRef<THREE.PerspectiveCamera>(null);

    useFrame((state, delta) => {
        if (!cameraRef.current) return;

        if (cinematicRef) {
            const { camera } = cinematicRef;

            // Apply camera angle
            let targetY = 1.6;  // Eye level
            if (camera.angle === 'low') targetY = 0.5;
            if (camera.angle === 'high') targetY = 5.0;

            // Smooth transition
            cameraRef.current.position.y = THREE.MathUtils.lerp(
                cameraRef.current.position.y,
                targetY,
                delta * 2.0
            );

            // Dutch angle (tilted) based on consciousness level
            const tilt = consciousnessLevel * 0.3;  // Up to 0.3 radians (~17 degrees)
            cameraRef.current.rotation.z = THREE.MathUtils.lerp(
                cameraRef.current.rotation.z,
                tilt,
                delta
            );

            // Camera shake during ego dissolution
            if (consciousnessLevel > 0.7) {
                const shake = (consciousnessLevel - 0.7) * 0.1;
                cameraRef.current.position.x += (Math.random() - 0.5) * shake;
                cameraRef.current.position.y += (Math.random() - 0.5) * shake;
            }
        }
    });

    return (
        <PerspectiveCamera
            ref={cameraRef}
            makeDefault
            position={[0, 1.6, 5]}
            fov={cinematicRef?.camera.focal_length ?
                (35 / cinematicRef.camera.focal_length) * 50 :
                75}
        />
    );
};

// ============================================================================
// MAIN COMPONENT
// ============================================================================

/**
 * Complete dream visualization scene
 */
const DreamScene: React.FC<DreamVisualizerProps> = ({
    dreamScene,
    participants,
    isPaused = false,
    playbackSpeed = 1.0,
    activePerspective
}) => {
    const [elapsedTime, setElapsedTime] = useState(0);

    useFrame((state, delta) => {
        if (!isPaused) {
            setElapsedTime(prev => prev + delta * playbackSpeed);
        }
    });

    // Get active participant's emotional state
    const activeParticipant = participants.find(p => p.resident_id === activePerspective) || participants[0];
    const emotionalState = activeParticipant.emotional_state;

    // Get primary cinematic reference (first symbol with references)
    const cinematicRef = dreamScene.symbols
        .find(s => s.cinematic_references && s.cinematic_references.length > 0)
        ?.cinematic_references?.[0];

    return (
        <>
            {/* Cinematic camera */}
            <CinematicCamera
                cinematicRef={cinematicRef}
                consciousnessLevel={dreamScene.consciousness_level}
            />

            {/* Atmospheric effects */}
            <Atmosphere
                setting={dreamScene.setting}
                cinematicRef={cinematicRef}
                consciousnessLevel={dreamScene.consciousness_level}
            />

            {/* Lighting based on cinematic reference */}
            {cinematicRef && (
                <>
                    <directionalLight
                        position={[5, 10, 5]}
                        intensity={cinematicRef.lighting.intensity}
                        color={new THREE.Color().setColorTemperature(cinematicRef.lighting.color_temp)}
                        castShadow
                    />
                    <ambientLight intensity={0.2} />
                </>
            )}

            {/* Default lighting if no cinematic reference */}
            {!cinematicRef && (
                <>
                    <directionalLight position={[5, 10, 5]} intensity={1} castShadow />
                    <ambientLight intensity={0.3} />
                </>
            )}

            {/* Render all symbols */}
            {dreamScene.symbols.map((symbol, idx) => (
                <SymbolObject
                    key={symbol.symbol_id}
                    symbol={symbol}
                    position={{ x: idx * 3 - 5, y: 0, z: 0 }}
                    emotionalState={emotionalState}
                    consciousnessLevel={dreamScene.consciousness_level}
                />
            ))}

            {/* Ground plane */}
            <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, -0.5, 0]} receiveShadow>
                <planeGeometry args={[100, 100]} />
                <meshStandardMaterial color="#333333" />
            </mesh>

            {/* Post-processing for cinematic look */}
            <EffectComposer>
                <Bloom intensity={0.5} luminanceThreshold={0.9} />
                <DepthOfField
                    focusDistance={0.01}
                    focalLength={cinematicRef?.camera.focal_length ? cinematicRef.camera.focal_length / 1000 : 0.05}
                    bokehScale={2}
                />
                <Vignette darkness={0.5} />
            </EffectComposer>

            {/* Debug stats */}
            <Stats />
        </>
    );
};

/**
 * Main Dream Visualizer Component
 */
export const DreamVisualizer: React.FC<DreamVisualizerProps> = (props) => {
    return (
        <div style={{ width: '100%', height: '100vh', position: 'relative' }}>
            <Canvas shadows dpr={[1, 2]} gl={{ antialias: true }}>
                <DreamScene {...props} />
                <OrbitControls enablePan={false} minDistance={2} maxDistance={20} />
            </Canvas>

            {/* Overlay info */}
            <div style={{
                position: 'absolute',
                top: 20,
                left: 20,
                color: 'white',
                fontFamily: 'monospace',
                fontSize: '14px',
                background: 'rgba(0,0,0,0.7)',
                padding: '10px',
                borderRadius: '5px'
            }}>
                <div><strong>{props.dreamScene.title}</strong></div>
                <div>Consciousness: {(props.dreamScene.consciousness_level * 100).toFixed(0)}%</div>
                <div>Participants: {props.participants.length}</div>
                <div>Symbols: {props.dreamScene.symbols.length}</div>
            </div>
        </div>
    );
};

export default DreamVisualizer;
