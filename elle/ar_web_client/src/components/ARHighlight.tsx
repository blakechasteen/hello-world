/**
 * ARHighlight Component - Visual highlight around AR objects
 *
 * Renders:
 * - Bounding box
 * - Glow effect
 * - Pulse animation
 *
 * Created: 2025-11-22
 */

import { useRef } from 'react'
import { useFrame } from '@react-three/fiber'
import * as THREE from 'three'

interface ARHighlightProps {
  id: string
  targetObjectId: string
  highlightType?: string
  intensity?: number
  style: {
    color?: string
    opacity?: number
    animation?: string
  }
}

export default function ARHighlight({
  highlightType = 'bounding_box',
  intensity = 1.0,
  style,
}: ARHighlightProps) {
  const boxRef = useRef<THREE.Mesh>(null)
  const timeRef = useRef(0)

  // Pulse animation
  useFrame((state, delta) => {
    if (!boxRef.current) return

    timeRef.current += delta

    if (style.animation === 'pulse') {
      // Pulsing scale
      const scale = 1.0 + Math.sin(timeRef.current * 3) * 0.1
      boxRef.current.scale.setScalar(scale)
    }
  })

  const color = style.color || '#FFD700'
  const opacity = (style.opacity || 0.5) * intensity

  return (
    <group position={[0, 0.5, -1.5]}>
      {highlightType === 'bounding_box' && (
        <mesh ref={boxRef}>
          <boxGeometry args={[0.5, 0.5, 0.5]} />
          <meshBasicMaterial
            color={color}
            transparent
            opacity={opacity}
            wireframe
          />
        </mesh>
      )}

      {highlightType === 'glow' && (
        <mesh ref={boxRef}>
          <sphereGeometry args={[0.3, 16, 16]} />
          <meshBasicMaterial
            color={color}
            transparent
            opacity={opacity * 0.3}
            side={THREE.BackSide}
          />
        </mesh>
      )}
    </group>
  )
}
