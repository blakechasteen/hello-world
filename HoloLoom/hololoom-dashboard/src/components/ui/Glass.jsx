import React from 'react';
import './Glass.css';

/**
 * Base Glass Component
 * Provides the core glassmorphic effect with configurable intensity and border.
 */
export const GlassPanel = ({
    children,
    className = '',
    intensity = 'medium', // low, medium, high
    interactive = false,
    ...props
}) => {
    return (
        <div
            className={`glass-panel glass-${intensity} ${interactive ? 'glass-interactive' : ''} ${className}`}
            {...props}
        >
            {children}
        </div>
    );
};

/**
 * Glass Button
 * A button with glass styling and hover effects.
 */
export const GlassButton = ({
    children,
    variant = 'primary', // primary, secondary, danger
    className = '',
    ...props
}) => {
    return (
        <button
            className={`glass-button variant-${variant} ${className}`}
            {...props}
        >
            {children}
        </button>
    );
};

/**
 * Glass Input
 * An input field with subtle glass background.
 */
export const GlassInput = ({ className = '', ...props }) => {
    return (
        <input
            className={`glass-input ${className}`}
            {...props}
        />
    );
};
