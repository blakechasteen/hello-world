/**
 * Dream Controls - UI controls for dream playback and perspective switching
 *
 * Controls:
 * - Pause/play dream
 * - Speed control (0.5x, 1x, 2x)
 * - Perspective switching (for shared dreams)
 * - Exit dream
 * - Consciousness slider (for testing)
 */

import React, { useState } from 'react';

export interface DreamControlsProps {
    isPaused: boolean;
    playbackSpeed: number;
    activePerspective: string;
    participants: Array<{ resident_id: string; name: string }>;
    consciousnessLevel: number;
    onPauseToggle: () => void;
    onSpeedChange: (speed: number) => void;
    onPerspectiveChange: (residentId: string) => void;
    onConsciousnessChange?: (level: number) => void;
    onExit: () => void;
}

export const DreamControls: React.FC<DreamControlsProps> = ({
    isPaused,
    playbackSpeed,
    activePerspective,
    participants,
    consciousnessLevel,
    onPauseToggle,
    onSpeedChange,
    onPerspectiveChange,
    onConsciousnessChange,
    onExit
}) => {
    const [showAdvanced, setShowAdvanced] = useState(false);

    const speeds = [0.5, 1.0, 2.0];

    return (
        <div style={{
            position: 'absolute',
            bottom: 20,
            left: '50%',
            transform: 'translateX(-50%)',
            background: 'rgba(0, 0, 0, 0.85)',
            padding: '15px 20px',
            borderRadius: '12px',
            color: 'white',
            fontFamily: 'system-ui, -apple-system, sans-serif',
            minWidth: '400px',
            maxWidth: '600px',
            backdropFilter: 'blur(10px)',
            border: '1px solid rgba(255, 255, 255, 0.1)',
            boxShadow: '0 8px 32px rgba(0, 0, 0, 0.5)'
        }}>
            {/* Main controls row */}
            <div style={{
                display: 'flex',
                alignItems: 'center',
                gap: '15px',
                marginBottom: '10px'
            }}>
                {/* Pause/Play button */}
                <button
                    onClick={onPauseToggle}
                    style={{
                        background: isPaused ? '#4CAF50' : '#FF9800',
                        border: 'none',
                        color: 'white',
                        padding: '10px 20px',
                        borderRadius: '6px',
                        cursor: 'pointer',
                        fontSize: '14px',
                        fontWeight: 'bold',
                        transition: 'all 0.2s',
                        display: 'flex',
                        alignItems: 'center',
                        gap: '8px'
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.opacity = '0.8'}
                    onMouseLeave={(e) => e.currentTarget.style.opacity = '1'}
                >
                    {isPaused ? (
                        <>
                            <span style={{ fontSize: '16px' }}>▶</span>
                            Play
                        </>
                    ) : (
                        <>
                            <span style={{ fontSize: '16px' }}>⏸</span>
                            Pause
                        </>
                    )}
                </button>

                {/* Speed control */}
                <div style={{ display: 'flex', gap: '5px' }}>
                    {speeds.map(speed => (
                        <button
                            key={speed}
                            onClick={() => onSpeedChange(speed)}
                            style={{
                                background: playbackSpeed === speed ? '#2196F3' : 'rgba(255, 255, 255, 0.1)',
                                border: 'none',
                                color: 'white',
                                padding: '8px 12px',
                                borderRadius: '4px',
                                cursor: 'pointer',
                                fontSize: '13px',
                                transition: 'all 0.2s'
                            }}
                            onMouseEnter={(e) => e.currentTarget.style.background =
                                playbackSpeed === speed ? '#1976D2' : 'rgba(255, 255, 255, 0.2)'}
                            onMouseLeave={(e) => e.currentTarget.style.background =
                                playbackSpeed === speed ? '#2196F3' : 'rgba(255, 255, 255, 0.1)'}
                        >
                            {speed}×
                        </button>
                    ))}
                </div>

                {/* Perspective selector */}
                <select
                    value={activePerspective}
                    onChange={(e) => onPerspectiveChange(e.target.value)}
                    style={{
                        background: 'rgba(255, 255, 255, 0.1)',
                        border: '1px solid rgba(255, 255, 255, 0.2)',
                        color: 'white',
                        padding: '8px 12px',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        fontSize: '13px',
                        flex: 1,
                        minWidth: '120px'
                    }}
                >
                    {participants.map(p => (
                        <option key={p.resident_id} value={p.resident_id} style={{ background: '#222' }}>
                            {p.name}'s POV
                        </option>
                    ))}
                </select>

                {/* Advanced toggle */}
                <button
                    onClick={() => setShowAdvanced(!showAdvanced)}
                    style={{
                        background: 'rgba(255, 255, 255, 0.1)',
                        border: '1px solid rgba(255, 255, 255, 0.2)',
                        color: 'white',
                        padding: '8px 12px',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        fontSize: '13px',
                        transition: 'all 0.2s'
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.background = 'rgba(255, 255, 255, 0.2)'}
                    onMouseLeave={(e) => e.currentTarget.style.background = 'rgba(255, 255, 255, 0.1)'}
                >
                    {showAdvanced ? '▼' : '▶'} Advanced
                </button>

                {/* Exit button */}
                <button
                    onClick={onExit}
                    style={{
                        background: '#f44336',
                        border: 'none',
                        color: 'white',
                        padding: '8px 16px',
                        borderRadius: '4px',
                        cursor: 'pointer',
                        fontSize: '13px',
                        fontWeight: 'bold',
                        transition: 'all 0.2s'
                    }}
                    onMouseEnter={(e) => e.currentTarget.style.background = '#d32f2f'}
                    onMouseLeave={(e) => e.currentTarget.style.background = '#f44336'}
                >
                    Exit Dream
                </button>
            </div>

            {/* Advanced controls (collapsible) */}
            {showAdvanced && onConsciousnessChange && (
                <div style={{
                    paddingTop: '15px',
                    borderTop: '1px solid rgba(255, 255, 255, 0.1)',
                    marginTop: '5px'
                }}>
                    {/* Consciousness slider */}
                    <div style={{ marginBottom: '15px' }}>
                        <div style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            marginBottom: '8px',
                            fontSize: '13px'
                        }}>
                            <label>Consciousness Level (Ego Dissolution)</label>
                            <span style={{ color: '#4CAF50', fontWeight: 'bold' }}>
                                {(consciousnessLevel * 100).toFixed(0)}%
                            </span>
                        </div>
                        <input
                            type="range"
                            min="0"
                            max="100"
                            value={consciousnessLevel * 100}
                            onChange={(e) => onConsciousnessChange(parseInt(e.target.value) / 100)}
                            style={{
                                width: '100%',
                                accentColor: '#4CAF50',
                                cursor: 'pointer'
                            }}
                        />
                        <div style={{
                            display: 'flex',
                            justifyContent: 'space-between',
                            fontSize: '11px',
                            color: 'rgba(255, 255, 255, 0.6)',
                            marginTop: '4px'
                        }}>
                            <span>Ego Intact</span>
                            <span>Ego Dissolved</span>
                        </div>
                    </div>

                    {/* Visual feedback indicators */}
                    <div style={{
                        display: 'grid',
                        gridTemplateColumns: '1fr 1fr',
                        gap: '10px',
                        fontSize: '12px'
                    }}>
                        <div style={{
                            background: 'rgba(255, 255, 255, 0.05)',
                            padding: '8px',
                            borderRadius: '4px',
                            border: '1px solid rgba(255, 255, 255, 0.1)'
                        }}>
                            <div style={{ color: 'rgba(255, 255, 255, 0.6)', marginBottom: '4px' }}>
                                Camera Effect
                            </div>
                            <div style={{ fontWeight: 'bold' }}>
                                {consciousnessLevel < 0.3 ? 'Stable' :
                                 consciousnessLevel < 0.7 ? 'Tilted' :
                                 'Shaking'}
                            </div>
                        </div>

                        <div style={{
                            background: 'rgba(255, 255, 255, 0.05)',
                            padding: '8px',
                            borderRadius: '4px',
                            border: '1px solid rgba(255, 255, 255, 0.1)'
                        }}>
                            <div style={{ color: 'rgba(255, 255, 255, 0.6)', marginBottom: '4px' }}>
                                Symbol Behavior
                            </div>
                            <div style={{ fontWeight: 'bold' }}>
                                {consciousnessLevel < 0.5 ? 'Realistic' : 'Metaphorical'}
                            </div>
                        </div>
                    </div>

                    {/* Help text */}
                    <div style={{
                        marginTop: '10px',
                        fontSize: '11px',
                        color: 'rgba(255, 255, 255, 0.5)',
                        fontStyle: 'italic',
                        textAlign: 'center'
                    }}>
                        Higher consciousness = more symbolic/metaphorical dream experience
                    </div>
                </div>
            )}

            {/* Keyboard shortcuts hint */}
            <div style={{
                marginTop: '10px',
                paddingTop: '10px',
                borderTop: '1px solid rgba(255, 255, 255, 0.1)',
                fontSize: '11px',
                color: 'rgba(255, 255, 255, 0.5)',
                display: 'flex',
                justifyContent: 'space-between'
            }}>
                <span><kbd style={{ background: 'rgba(255,255,255,0.1)', padding: '2px 6px', borderRadius: '3px' }}>Space</kbd> Pause/Play</span>
                <span><kbd style={{ background: 'rgba(255,255,255,0.1)', padding: '2px 6px', borderRadius: '3px' }}>1-3</kbd> Speed</span>
                <span><kbd style={{ background: 'rgba(255,255,255,0.1)', padding: '2px 6px', borderRadius: '3px' }}>Esc</kbd> Exit</span>
            </div>
        </div>
    );
};

/**
 * Keyboard shortcuts hook
 */
export const useDreamKeyboardShortcuts = (
    onPauseToggle: () => void,
    onSpeedChange: (speed: number) => void,
    onExit: () => void
) => {
    React.useEffect(() => {
        const handleKeyDown = (e: KeyboardEvent) => {
            switch (e.key) {
                case ' ':
                    e.preventDefault();
                    onPauseToggle();
                    break;
                case '1':
                    onSpeedChange(0.5);
                    break;
                case '2':
                    onSpeedChange(1.0);
                    break;
                case '3':
                    onSpeedChange(2.0);
                    break;
                case 'Escape':
                    onExit();
                    break;
            }
        };

        window.addEventListener('keydown', handleKeyDown);
        return () => window.removeEventListener('keydown', handleKeyDown);
    }, [onPauseToggle, onSpeedChange, onExit]);
};

export default DreamControls;
