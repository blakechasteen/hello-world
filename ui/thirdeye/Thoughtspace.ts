/**
 * Thoughtspace - Full-screen 3D concept world
 *
 * Main scene manager for the ThirdEye visualization.
 * Integrates Three.js scene, ChatBridge, and TransitionEngine.
 *
 * Created: 2025-11-30
 * Author: HoloLoom Team
 */

import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import gsap from 'gsap';
import { ChatBridge, getChatBridge, Concept, ConceptExtraction, RenderCommand } from './ChatBridge';
import { TransitionEngine, ConceptNode } from './TransitionEngine';

// Display modes
export type ThoughtspaceMode = 'compact' | 'expanded' | 'fullscreen' | 'overlay';

// Configuration
export interface ThoughtspaceConfig {
  container: HTMLElement;
  chatServerUrl?: string;
  backgroundColor?: string;
  showFloorGrid?: boolean;
  enableShadows?: boolean;
  initialMode?: ThoughtspaceMode;
}

/**
 * Thoughtspace - Main 3D visualization manager
 */
export class Thoughtspace {
  // Three.js components
  private scene: THREE.Scene;
  private camera: THREE.PerspectiveCamera;
  private renderer: THREE.WebGLRenderer;
  private controls: OrbitControls;

  // Engines
  private chatBridge: ChatBridge;
  private transitionEngine: TransitionEngine;

  // State
  private mode: ThoughtspaceMode;
  private container: HTMLElement;
  private animationFrameId: number | null = null;
  private currentConcept: Concept | null = null;
  private isRunning: boolean = false;

  // Lights
  private ambientLight: THREE.AmbientLight;
  private directionalLight: THREE.DirectionalLight;

  // Floor grid
  private floorGrid: THREE.GridHelper | null = null;

  constructor(config: ThoughtspaceConfig) {
    this.container = config.container;
    this.mode = config.initialMode || 'compact';

    // Initialize Three.js scene
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(config.backgroundColor || '#f8fafc');

    // Fog for depth
    this.scene.fog = new THREE.Fog(config.backgroundColor || '#f8fafc', 50, 100);

    // Camera
    this.camera = new THREE.PerspectiveCamera(
      60,
      this.container.clientWidth / this.container.clientHeight,
      0.1,
      1000
    );
    this.camera.position.set(0, 5, 15);
    this.camera.lookAt(0, 0, 0);

    // Renderer
    this.renderer = new THREE.WebGLRenderer({
      antialias: true,
      alpha: true,
    });
    this.renderer.setSize(this.container.clientWidth, this.container.clientHeight);
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.shadowMap.enabled = config.enableShadows !== false;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.container.appendChild(this.renderer.domElement);

    // Controls
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.dampingFactor = 0.05;
    this.controls.maxDistance = 100;
    this.controls.minDistance = 5;

    // Lighting
    this.ambientLight = new THREE.AmbientLight(0xffffff, 0.6);
    this.scene.add(this.ambientLight);

    this.directionalLight = new THREE.DirectionalLight(0xffffff, 0.8);
    this.directionalLight.position.set(10, 20, 10);
    this.directionalLight.castShadow = config.enableShadows !== false;
    this.scene.add(this.directionalLight);

    // Floor grid
    if (config.showFloorGrid !== false) {
      this.floorGrid = new THREE.GridHelper(100, 20, 0xe2e8f0, 0xe2e8f0);
      this.floorGrid.position.y = -5;
      (this.floorGrid.material as THREE.Material).opacity = 0.3;
      (this.floorGrid.material as THREE.Material).transparent = true;
      this.scene.add(this.floorGrid);
    }

    // Engines
    this.chatBridge = getChatBridge(config.chatServerUrl);
    this.transitionEngine = new TransitionEngine(this.scene, this.camera);

    // Event handlers
    this.setupEventHandlers();
    this.setupResizeHandler();
  }

  /**
   * Setup chat bridge event handlers
   */
  private setupEventHandlers(): void {
    this.chatBridge.onConceptExtraction((extraction) => {
      this.handleConceptExtraction(extraction);
    });

    this.chatBridge.onRenderCommand((command) => {
      this.handleRenderCommand(command);
    });

    this.chatBridge.onConnectionChange((connected) => {
      console.log('[Thoughtspace] Connection:', connected ? 'open' : 'closed');
    });

    this.chatBridge.onErrorEvent((error) => {
      console.error('[Thoughtspace] Error:', error);
    });
  }

  /**
   * Setup resize handler
   */
  private setupResizeHandler(): void {
    const resizeObserver = new ResizeObserver(() => {
      this.resize();
    });
    resizeObserver.observe(this.container);
  }

  /**
   * Handle incoming concept extraction
   */
  private async handleConceptExtraction(extraction: ConceptExtraction): Promise<void> {
    const { concepts, transition, primary_concept } = extraction;

    if (primary_concept && transition) {
      // Animate transition to new concept
      await this.transitionEngine.transition(
        this.currentConcept,
        primary_concept,
        transition
      );
      this.currentConcept = primary_concept;
    } else if (primary_concept) {
      // No transition - just create the concept
      this.transitionEngine.createConceptNode(primary_concept);
      this.currentConcept = primary_concept;
    }

    // Add any additional concepts from the extraction
    for (const concept of concepts) {
      if (concept.id !== primary_concept?.id) {
        this.transitionEngine.createConceptNode(concept);
      }
    }

    // Focus camera on primary concept
    if (primary_concept) {
      this.focusConcept(primary_concept.id);
    }
  }

  /**
   * Handle render command from server
   */
  private handleRenderCommand(command: RenderCommand): void {
    switch (command.type) {
      case 'set_camera':
        this.setCameraFromCommand(command);
        break;

      case 'set_background':
        this.setBackgroundFromCommand(command);
        break;

      case 'remove_concept':
        if (command.target) {
          this.transitionEngine.removeNode(command.target);
        }
        break;

      default:
        console.log('[Thoughtspace] Unknown command:', command.type);
    }
  }

  /**
   * Set camera position from command
   */
  private setCameraFromCommand(command: RenderCommand): void {
    const data = command.data as {
      position?: { x: number; y: number; z: number };
      target?: { x: number; y: number; z: number };
      animate?: boolean;
      duration_ms?: number;
    };

    const duration = (data.duration_ms || 500) / 1000;

    if (data.animate !== false) {
      if (data.position) {
        gsap.to(this.camera.position, {
          x: data.position.x,
          y: data.position.y,
          z: data.position.z,
          duration,
          ease: 'power2.inOut',
        });
      }
      if (data.target) {
        gsap.to(this.controls.target, {
          x: data.target.x,
          y: data.target.y,
          z: data.target.z,
          duration,
          ease: 'power2.inOut',
        });
      }
    } else {
      if (data.position) {
        this.camera.position.set(data.position.x, data.position.y, data.position.z);
      }
      if (data.target) {
        this.controls.target.set(data.target.x, data.target.y, data.target.z);
      }
    }
  }

  /**
   * Set background from command
   */
  private setBackgroundFromCommand(command: RenderCommand): void {
    const data = command.data as { color?: string };
    if (data.color) {
      this.scene.background = new THREE.Color(data.color);
      if (this.scene.fog) {
        (this.scene.fog as THREE.Fog).color = new THREE.Color(data.color);
      }
    }
  }

  /**
   * Focus camera on a concept
   */
  focusConcept(conceptId: string): void {
    const node = this.transitionEngine.getNode(conceptId);
    if (!node) return;

    const pos = node.concept.position;

    gsap.to(this.camera.position, {
      x: pos.x,
      y: pos.y + 5,
      z: pos.z + 15,
      duration: 0.5,
      ease: 'power2.inOut',
    });

    gsap.to(this.controls.target, {
      x: pos.x,
      y: pos.y,
      z: pos.z,
      duration: 0.5,
      ease: 'power2.inOut',
    });
  }

  /**
   * Toggle display mode
   */
  toggleMode(): void {
    switch (this.mode) {
      case 'compact':
        this.setMode('fullscreen');
        break;
      case 'fullscreen':
        this.setMode('compact');
        break;
      default:
        this.setMode('compact');
    }
  }

  /**
   * Set display mode
   */
  setMode(mode: ThoughtspaceMode): void {
    this.mode = mode;

    // Update container size based on mode
    switch (mode) {
      case 'compact':
        this.container.style.width = '300px';
        this.container.style.height = '100%';
        this.container.style.position = 'relative';
        break;

      case 'expanded':
        this.container.style.width = '50%';
        this.container.style.height = '100%';
        this.container.style.position = 'relative';
        break;

      case 'fullscreen':
        this.container.style.width = '100vw';
        this.container.style.height = '100vh';
        this.container.style.position = 'fixed';
        this.container.style.top = '0';
        this.container.style.left = '0';
        this.container.style.zIndex = '1000';
        break;

      case 'overlay':
        this.container.style.width = '100vw';
        this.container.style.height = '100vh';
        this.container.style.position = 'fixed';
        this.container.style.top = '0';
        this.container.style.left = '0';
        this.container.style.zIndex = '1000';
        this.container.style.pointerEvents = 'none';
        break;
    }

    this.resize();

    // Animate camera based on mode
    if (mode === 'fullscreen' || mode === 'overlay') {
      gsap.to(this.camera.position, {
        z: 25,
        y: 10,
        duration: 0.6,
        ease: 'power2.inOut',
      });
    } else {
      gsap.to(this.camera.position, {
        z: 15,
        y: 5,
        duration: 0.4,
        ease: 'power2.inOut',
      });
    }
  }

  /**
   * Resize renderer to container
   */
  resize(): void {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;

    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
    this.renderer.setSize(width, height);
  }

  /**
   * Start the render loop and connect to server
   */
  async start(): Promise<void> {
    if (this.isRunning) return;
    this.isRunning = true;

    // Connect to chat bridge
    try {
      await this.chatBridge.connect();
    } catch (error) {
      console.warn('[Thoughtspace] Could not connect to server, running in demo mode');
    }

    // Start render loop
    this.animate();
  }

  /**
   * Stop rendering and disconnect
   */
  stop(): void {
    this.isRunning = false;
    if (this.animationFrameId !== null) {
      cancelAnimationFrame(this.animationFrameId);
      this.animationFrameId = null;
    }
    this.chatBridge.disconnect();
  }

  /**
   * Main animation loop
   */
  private animate = (): void => {
    if (!this.isRunning) return;
    this.animationFrameId = requestAnimationFrame(this.animate);

    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  };

  /**
   * Add a concept programmatically
   */
  addConcept(concept: Concept): ConceptNode {
    return this.transitionEngine.createConceptNode(concept);
  }

  /**
   * Get current mode
   */
  getMode(): ThoughtspaceMode {
    return this.mode;
  }

  /**
   * Get all concept nodes
   */
  getNodes(): ConceptNode[] {
    return this.transitionEngine.getAllNodes();
  }

  /**
   * Cleanup resources
   */
  dispose(): void {
    this.stop();
    this.transitionEngine.dispose();
    this.renderer.dispose();
    this.container.removeChild(this.renderer.domElement);
  }
}

/**
 * Create and initialize a Thoughtspace instance
 */
export function createThoughtspace(config: ThoughtspaceConfig): Thoughtspace {
  return new Thoughtspace(config);
}
