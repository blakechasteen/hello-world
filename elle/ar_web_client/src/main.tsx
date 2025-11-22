/**
 * Main Entry Point - Elle AR Client
 *
 * Initializes React Three Fiber AR application with WebXR support.
 *
 * Created: 2025-11-22 (Phase 1 Prototype)
 */

import React from 'react'
import ReactDOM from 'react-dom/client'
import App from './App'
import './index.css'

// Check for WebXR support
const checkWebXRSupport = async (): Promise<boolean> => {
  if (!('xr' in navigator)) {
    return false
  }

  try {
    const isSupported = await navigator.xr?.isSessionSupported('immersive-ar')
    return isSupported || false
  } catch (error) {
    console.error('WebXR check failed:', error)
    return false
  }
}

// Initialize app
const init = async () => {
  const root = ReactDOM.createRoot(document.getElementById('root')!)

  const arSupported = await checkWebXRSupport()

  if (!arSupported) {
    // Show fallback UI for non-AR devices
    root.render(
      <React.StrictMode>
        <div className="ar-not-supported">
          <h2>AR Not Available</h2>
          <p>This device doesn't support WebXR AR.</p>
          <p>For the best experience, try:</p>
          <ul style={{ textAlign: 'left', marginTop: '1rem', color: '#999' }}>
            <li>Meta Quest 3 browser</li>
            <li>Chrome on Android (ARCore devices)</li>
            <li>Safari on iPhone/iPad (iOS 15+)</li>
          </ul>
          <p style={{ marginTop: '1rem', fontSize: '0.875rem' }}>
            Desktop mode available for development testing.
          </p>
        </div>
      </React.StrictMode>
    )
    return
  }

  // Render AR app
  root.render(
    <React.StrictMode>
      <App />
    </React.StrictMode>
  )
}

// Start
init().catch(console.error)

// Enable hot module replacement for development
if (import.meta.hot) {
  import.meta.hot.accept()
}
