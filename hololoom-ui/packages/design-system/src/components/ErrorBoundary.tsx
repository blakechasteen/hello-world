'use client';

import React, { Component, type ErrorInfo, type ReactNode } from 'react';

export interface ErrorBoundaryProps {
  children: ReactNode;
  /** Custom fallback UI. Receives error and reset function. */
  fallback?: (props: { error: Error; resetError: () => void }) => ReactNode;
}

interface ErrorBoundaryState {
  error: Error | null;
}

export class ErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props);
    this.state = { error: null };
  }

  static getDerivedStateFromError(error: Error): ErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('[HoloLoom] Component error:', error, errorInfo);
  }

  resetError = () => {
    this.setState({ error: null });
  };

  render() {
    if (this.state.error) {
      if (this.props.fallback) {
        return this.props.fallback({
          error: this.state.error,
          resetError: this.resetError,
        });
      }

      return (
        <div
          role="alert"
          className="flex flex-col items-center justify-center p-8 text-center"
        >
          <div className="w-12 h-12 rounded-full bg-safety-danger/10 flex items-center justify-center mb-4">
            <svg
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              className="text-safety-danger"
            >
              <circle cx="12" cy="12" r="10" />
              <path d="M12 8v4" />
              <path d="M12 16h.01" />
            </svg>
          </div>
          <h3 className="text-lg font-medium text-fg-primary mb-2">
            Something went wrong
          </h3>
          <p className="text-sm text-fg-tertiary mb-4 max-w-sm">
            An unexpected error occurred. Try refreshing the page or click below to retry.
          </p>
          <button
            onClick={this.resetError}
            className="px-4 py-2 text-sm font-medium rounded-lg bg-cosmic-nebula text-white hover:bg-cosmic-nebula/90 transition-colors focus-visible:ring-2 focus-visible:ring-cosmic-nebula focus-visible:ring-offset-2"
          >
            Try again
          </button>
        </div>
      );
    }

    return this.props.children;
  }
}
