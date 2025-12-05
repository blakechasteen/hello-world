/**
 * AROverlay Component - Text/icon overlay on AR objects
 *
 * Renders a text label attached to a 3D object in AR space.
 * Billboard effect: always faces the camera.
 *
 * Created: 2025-11-22
 */

import { useRef, useMemo } from 'react'
import { useFrame } from '@react-three/fiber'
import { Text } from '@react-three/drei'
import * as THREE from 'three'

interface AROverlayProps {
  id: string
  targetObjectId?: string
  content: string
  position?: { x: number; y: number; z: number }
  positionOffset?: { x: number; y: number; z: number }
  faceUser?: boolean
  style: {
    color?: string
    opacity?: number
    size?: number
  }
}

export default function AROverlay({
  content,
  position,
  positionOffset = { x: 0, y: 0.2, z: 0 },
  faceUser = true,
  style,
}: AROverlayProps) {
  const textRef = useRef<any>(null)

  // Calculate position
  const finalPosition = useMemo(() => {
    const base = position || { x: 0, y: 1.5, z: -1 }
    return new THREE.Vector3(
      base.x + positionOffset.x,
      base.y + positionOffset.y,
      base.z + positionOffset.z
    )
  }, [position, positionOffset])

  // Billboard effect - always face camera
  useFrame(({ camera }) => {
    if (textRef.current && faceUser) {
      textRef.current.quaternion.copy(camera.quaternion)
    }
  })

  return (
    <Text
      ref={textRef}
      position={finalPosition}
      fontSize={0.12 * (style.size || 1)}
      color={style.color || '#ffffff'}
      anchorX="center"
      anchorY="middle"
      outlineWidth={0.002}
      outlineColor="#000000"
    >
      {content}
    </Text>
  )
}
