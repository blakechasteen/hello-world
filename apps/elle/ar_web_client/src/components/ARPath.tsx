/**
 * ARPath Component - 3D path visualization for navigation
 *
 * Renders:
 * - Arrow pointing to destination
 * - Dotted line path
 * - Distance marker
 *
 * Created: 2025-11-22
 */

import { useMemo } from 'react'
import * as THREE from 'three'

interface ARPathProps {
  id: string
  waypoints: Array<{ x: number; y: number; z: number }>
  pathType?: string
  width?: number
  endMarker?: boolean
  style: {
    color?: string
    opacity?: number
  }
}

export default function ARPath({
  waypoints,
  pathType = 'arrow',
  width = 0.05,
  endMarker = true,
  style,
}: ARPathProps) {
  const color = style.color || '#667eea'
  const opacity = style.opacity || 0.8

  // Convert waypoints to Vector3 array
  const points = useMemo(
    () => waypoints.map((wp) => new THREE.Vector3(wp.x, wp.y, wp.z)),
    [waypoints]
  )

  // Create path curve
  const curve = useMemo(() => {
    if (points.length < 2) return null
    return new THREE.CatmullRomCurve3(points)
  }, [points])

  if (!curve) return null

  const endPoint = points[points.length - 1]

  return (
    <group>
      {/* Path line */}
      {pathType === 'line' && (
        <line>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={points.length}
              array={new Float32Array(points.flatMap((p) => [p.x, p.y, p.z]))}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial color={color} transparent opacity={opacity} linewidth={width * 100} />
        </line>
      )}

      {/* Arrow */}
      {pathType === 'arrow' && (
        <arrowHelper
          args={[
            points[points.length - 1].clone().sub(points[0]).normalize(),
            points[0],
            points[0].distanceTo(points[points.length - 1]),
            color,
            0.2,
            0.1,
          ]}
        />
      )}

      {/* End marker */}
      {endMarker && (
        <mesh position={endPoint}>
          <sphereGeometry args={[0.1, 16, 16]} />
          <meshBasicMaterial color={color} transparent opacity={opacity} />
        </mesh>
      )}
    </group>
  )
}
