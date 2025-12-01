/**
 * TransitionEngine - Smooth animations between concepts
 *
 * Uses GSAP and Three.js to animate transitions between concept
 * visualizations based on semantic distance and transition type.
 *
 * Created: 2025-11-30
 * Author: HoloLoom Team
 */

import * as THREE from 'three';
import gsap from 'gsap';
import { Concept, Transition } from './ChatBridge';

// Easing functions (GSAP uses these names)
const EASINGS: Record<string, string> = {
  easeInOutQuad: 'power2.inOut',
  easeInOutCubic: 'power3.inOut',
  easeInOutQuart: 'power4.inOut',
  easeInOutSine: 'sine.inOut',
  easeIn: 'power2.in',
  easeOut: 'power2.out',
  easeOutBack: 'back.out(1.7)',
  easeInCubic: 'power3.in',
  easeOutCubic: 'power3.out',
};

// Convert our easing name to GSAP
function toGsapEasing(easing: string): string {
  return EASINGS[easing] || 'power2.inOut';
}

/**
 * Concept node in Three.js scene
 */
export interface ConceptNode {
  group: THREE.Group;
  mesh: THREE.Mesh;
  label: THREE.Sprite | null;
  concept: Concept;
  edges: THREE.Line[];
}

/**
 * TransitionEngine handles all concept animations
 */
export class TransitionEngine {
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private nodes: Map<string, ConceptNode> = new Map();
  private timeline: gsap.core.Timeline | null = null;

  constructor(scene: THREE.Scene, camera: THREE.PerspectiveCamera) {
    this.scene = scene;
    this.camera = camera;
  }

  /**
   * Create a visual node for a concept
   */
  createConceptNode(concept: Concept): ConceptNode {
    // Create group to hold mesh and label
    const group = new THREE.Group();
    group.position.set(
      concept.position.x,
      concept.position.y,
      concept.position.z
    );
    group.userData.concept = concept;

    // Create sphere mesh
    const geometry = new THREE.SphereGeometry(
      0.5 + concept.importance * 1.5,  // Size based on importance
      32, 32
    );
    const material = new THREE.MeshPhongMaterial({
      color: this.getConceptColor(concept.type),
      opacity: 0.7 + concept.confidence * 0.3,
      transparent: true,
      emissive: this.getConceptColor(concept.type),
      emissiveIntensity: 0.1,
    });
    const mesh = new THREE.Mesh(geometry, material);
    group.add(mesh);

    // Create label sprite
    const label = this.createLabel(concept.label);
    if (label) {
      label.position.y = 1 + concept.importance;
      group.add(label);
    }

    this.scene.add(group);

    const node: ConceptNode = {
      group,
      mesh,
      label,
      concept,
      edges: [],
    };

    this.nodes.set(concept.id, node);
    return node;
  }

  /**
   * Get color for concept type
   */
  private getConceptColor(type: string): number {
    const colors: Record<string, number> = {
      entity: 0x3b82f6,        // Blue
      relationship: 0x10b981,   // Green
      process: 0x8b5cf6,        // Purple
      comparison: 0xf59e0b,     // Amber
      hierarchy: 0x6366f1,      // Indigo
      temporal: 0x06b6d4,       // Cyan
      probability: 0xec4899,    // Pink
      decision: 0xf97316,       // Orange
      activation: 0xeab308,     // Yellow
      memory: 0x14b8a6,         // Teal
      cluster: 0xa855f7,        // Purple
      thread: 0x64748b,         // Slate
      world: 0x0ea5e9,          // Sky
    };
    return colors[type.toLowerCase()] || 0x64748b;
  }

  /**
   * Create a text label sprite
   */
  private createLabel(text: string): THREE.Sprite | null {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    if (!ctx) return null;

    canvas.width = 256;
    canvas.height = 64;

    ctx.fillStyle = '#1e293b';
    ctx.font = '24px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText(text, 128, 40);

    const texture = new THREE.CanvasTexture(canvas);
    const material = new THREE.SpriteMaterial({
      map: texture,
      transparent: true,
    });
    const sprite = new THREE.Sprite(material);
    sprite.scale.set(4, 1, 1);

    return sprite;
  }

  /**
   * Execute a transition between concepts
   */
  async transition(
    fromConcept: Concept | null,
    toConcept: Concept,
    transition: Transition
  ): Promise<ConceptNode> {
    // Get or create the target node
    let toNode = this.nodes.get(toConcept.id);
    if (!toNode) {
      toNode = this.createConceptNode(toConcept);
      // Start invisible for fade-in
      toNode.group.scale.setScalar(0);
      (toNode.mesh.material as THREE.MeshPhongMaterial).opacity = 0;
    }

    const fromNode = fromConcept ? this.nodes.get(fromConcept.id) : null;
    const duration = transition.duration_ms / 1000;
    const easing = toGsapEasing(transition.easing);

    // Kill any existing timeline
    this.timeline?.kill();
    this.timeline = gsap.timeline();

    switch (transition.type) {
      case 'morph':
        await this.morphTransition(fromNode, toNode, duration, easing, transition);
        break;

      case 'blend':
        await this.blendTransition(fromNode, toNode, duration, easing);
        break;

      case 'dissolve_emerge':
        await this.dissolveEmergeTransition(fromNode, toNode, duration, easing, transition);
        break;

      case 'collapse_expand':
        await this.collapseExpandTransition(fromNode, toNode, duration, easing);
        break;

      case 'pulse':
        await this.pulseTransition(toNode, duration, easing, transition);
        break;

      case 'zoom':
        await this.zoomTransition(toNode, duration, easing);
        break;

      default:
        // Default: just fade in
        await this.fadeInTransition(toNode, duration, easing);
    }

    return toNode;
  }

  /**
   * Morph transition - smooth shape interpolation
   */
  private async morphTransition(
    fromNode: ConceptNode | null,
    toNode: ConceptNode,
    duration: number,
    easing: string,
    transition: Transition
  ): Promise<void> {
    if (fromNode) {
      // Move from position to new position
      this.timeline!.to(fromNode.group.position, {
        x: toNode.concept.position.x,
        y: toNode.concept.position.y,
        z: toNode.concept.position.z,
        duration,
        ease: easing,
      }, 0);

      // Scale pulse during morph
      this.timeline!.to(fromNode.group.scale, {
        x: transition.visual.scale_factor,
        y: transition.visual.scale_factor,
        z: transition.visual.scale_factor,
        duration: duration * 0.5,
        ease: 'power2.in',
        yoyo: true,
        repeat: 1,
      }, 0);

      // Update node data
      fromNode.concept = toNode.concept;

      // Fade out old, fade in new
      const fromMat = fromNode.mesh.material as THREE.MeshPhongMaterial;
      const toMat = toNode.mesh.material as THREE.MeshPhongMaterial;

      this.timeline!.to(fromMat, {
        opacity: 0,
        duration: duration * 0.5,
        ease: 'power2.in',
      }, 0);

      this.timeline!.to(toMat, {
        opacity: 0.7 + toNode.concept.confidence * 0.3,
        duration: duration * 0.5,
        ease: 'power2.out',
      }, duration * 0.3);
    }

    // Ensure new node is visible
    this.timeline!.to(toNode.group.scale, {
      x: 1, y: 1, z: 1,
      duration: duration * 0.5,
      ease: 'power2.out',
    }, duration * 0.3);

    return new Promise(resolve => {
      this.timeline!.eventCallback('onComplete', resolve);
    });
  }

  /**
   * Blend transition - crossfade
   */
  private async blendTransition(
    fromNode: ConceptNode | null,
    toNode: ConceptNode,
    duration: number,
    easing: string
  ): Promise<void> {
    if (fromNode) {
      const fromMat = fromNode.mesh.material as THREE.MeshPhongMaterial;
      this.timeline!.to(fromMat, {
        opacity: 0,
        duration: duration * 0.6,
        ease: easing,
      }, 0);
    }

    const toMat = toNode.mesh.material as THREE.MeshPhongMaterial;
    this.timeline!.fromTo(toMat,
      { opacity: 0 },
      { opacity: 0.7 + toNode.concept.confidence * 0.3, duration: duration * 0.6, ease: easing },
      duration * 0.4
    );

    this.timeline!.fromTo(toNode.group.scale,
      { x: 0, y: 0, z: 0 },
      { x: 1, y: 1, z: 1, duration: duration * 0.6, ease: easing },
      duration * 0.4
    );

    return new Promise(resolve => {
      this.timeline!.eventCallback('onComplete', resolve);
    });
  }

  /**
   * Dissolve/emerge transition - blur fade
   */
  private async dissolveEmergeTransition(
    fromNode: ConceptNode | null,
    toNode: ConceptNode,
    duration: number,
    easing: string,
    transition: Transition
  ): Promise<void> {
    if (fromNode) {
      const fromMat = fromNode.mesh.material as THREE.MeshPhongMaterial;

      // Fade out with scale
      this.timeline!.to(fromNode.group.scale, {
        x: 0.5, y: 0.5, z: 0.5,
        duration: duration * 0.4,
        ease: 'power2.in',
      }, 0);

      this.timeline!.to(fromMat, {
        opacity: 0,
        duration: duration * 0.4,
        ease: 'power2.in',
      }, 0);
    }

    // Emerge new
    const toMat = toNode.mesh.material as THREE.MeshPhongMaterial;
    const overlap = transition.visual.fade_overlap;

    this.timeline!.fromTo(toNode.group.scale,
      { x: 0.5, y: 0.5, z: 0.5 },
      { x: 1, y: 1, z: 1, duration: duration * 0.4, ease: 'power2.out' },
      duration * (1 - overlap)
    );

    this.timeline!.fromTo(toMat,
      { opacity: 0 },
      { opacity: 0.7 + toNode.concept.confidence * 0.3, duration: duration * 0.4, ease: 'power2.out' },
      duration * (1 - overlap)
    );

    return new Promise(resolve => {
      this.timeline!.eventCallback('onComplete', resolve);
    });
  }

  /**
   * Collapse/expand transition
   */
  private async collapseExpandTransition(
    fromNode: ConceptNode | null,
    toNode: ConceptNode,
    duration: number,
    easing: string
  ): Promise<void> {
    if (fromNode) {
      this.timeline!.to(fromNode.group.scale, {
        x: 0, y: 0, z: 0,
        duration: duration * 0.4,
        ease: 'power2.in',
      }, 0);
    }

    this.timeline!.fromTo(toNode.group.scale,
      { x: 0, y: 0, z: 0 },
      { x: 1, y: 1, z: 1, duration: duration * 0.4, ease: 'back.out(1.7)' },
      duration * 0.3
    );

    const toMat = toNode.mesh.material as THREE.MeshPhongMaterial;
    this.timeline!.fromTo(toMat,
      { opacity: 0 },
      { opacity: 0.7 + toNode.concept.confidence * 0.3, duration: duration * 0.3 },
      duration * 0.3
    );

    return new Promise(resolve => {
      this.timeline!.eventCallback('onComplete', resolve);
    });
  }

  /**
   * Pulse transition - emphasis without change
   */
  private async pulseTransition(
    node: ConceptNode,
    duration: number,
    easing: string,
    transition: Transition
  ): Promise<void> {
    const scale = transition.visual.scale_factor;

    this.timeline!.to(node.group.scale, {
      x: scale, y: scale, z: scale,
      duration: duration * 0.5,
      ease: 'power2.in',
      yoyo: true,
      repeat: 1,
    });

    // Glow effect
    const mat = node.mesh.material as THREE.MeshPhongMaterial;
    this.timeline!.to(mat, {
      emissiveIntensity: 0.5,
      duration: duration * 0.5,
      yoyo: true,
      repeat: 1,
    }, 0);

    return new Promise(resolve => {
      this.timeline!.eventCallback('onComplete', resolve);
    });
  }

  /**
   * Zoom transition - camera movement
   */
  private async zoomTransition(
    toNode: ConceptNode,
    duration: number,
    easing: string
  ): Promise<void> {
    const target = toNode.concept.position;

    this.timeline!.to(this.camera.position, {
      x: target.x,
      y: target.y + 5,
      z: target.z + 15,
      duration,
      ease: easing,
    });

    return new Promise(resolve => {
      this.timeline!.eventCallback('onComplete', resolve);
    });
  }

  /**
   * Simple fade in
   */
  private async fadeInTransition(
    node: ConceptNode,
    duration: number,
    easing: string
  ): Promise<void> {
    const mat = node.mesh.material as THREE.MeshPhongMaterial;

    this.timeline!.fromTo(mat,
      { opacity: 0 },
      { opacity: 0.7 + node.concept.confidence * 0.3, duration, ease: easing }
    );

    this.timeline!.fromTo(node.group.scale,
      { x: 0, y: 0, z: 0 },
      { x: 1, y: 1, z: 1, duration, ease: 'back.out(1.7)' },
      0
    );

    return new Promise(resolve => {
      this.timeline!.eventCallback('onComplete', resolve);
    });
  }

  /**
   * Remove a concept node
   */
  async removeNode(conceptId: string): Promise<void> {
    const node = this.nodes.get(conceptId);
    if (!node) return;

    await new Promise<void>(resolve => {
      gsap.to(node.group.scale, {
        x: 0, y: 0, z: 0,
        duration: 0.3,
        ease: 'power2.in',
        onComplete: () => {
          this.scene.remove(node.group);
          this.nodes.delete(conceptId);
          resolve();
        },
      });
    });
  }

  /**
   * Get a node by concept ID
   */
  getNode(conceptId: string): ConceptNode | undefined {
    return this.nodes.get(conceptId);
  }

  /**
   * Get all nodes
   */
  getAllNodes(): ConceptNode[] {
    return Array.from(this.nodes.values());
  }

  /**
   * Cleanup
   */
  dispose(): void {
    this.timeline?.kill();
    for (const node of this.nodes.values()) {
      this.scene.remove(node.group);
    }
    this.nodes.clear();
  }
}
