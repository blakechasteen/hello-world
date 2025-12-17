'use client';

import { useState } from 'react';
import Link from 'next/link';
import { Button, Badge, SafetyBadge } from '@hololoom/design-system';
import { useHoloLoom } from '@hololoom/api-client';

interface NavItem {
  label: string;
  href: string;
  badge?: string;
}

const navItems: NavItem[] = [
  { label: 'Chat', href: '/chat' },
  { label: 'Memory', href: '/memory' },
  { label: 'Workflows', href: '/workflow' },
  { label: 'Safety', href: '/safety', badge: 'Live' },
  { label: 'Analytics', href: '/analytics' },
];

export function Navigation() {
  const { isConnected, health } = useHoloLoom();
  const [mobileMenuOpen, setMobileMenuOpen] = useState(false);

  return (
    <nav className="sticky top-0 z-40 w-full border-b border-border-primary bg-bg-primary/80 backdrop-blur-lg">
      <div className="main-content">
        <div className="flex h-16 items-center justify-between">
          {/* Logo */}
          <Link href="/" className="flex items-center gap-3 group">
            <div className="relative">
              {/* Cosmic glow effect */}
              <div className="absolute inset-0 bg-cosmic-nebula/30 rounded-full blur-md group-hover:bg-cosmic-nebula/50 transition-colors" />
              <div className="relative w-8 h-8 rounded-full bg-gradient-to-br from-cosmic-nebula to-cosmic-aurora flex items-center justify-center">
                <span className="text-white font-bold text-sm">H</span>
              </div>
            </div>
            <span className="font-bold text-lg text-gradient-cosmic">HoloLoom</span>
          </Link>

          {/* Desktop Navigation */}
          <div className="hidden md:flex items-center gap-1">
            {navItems.map((item) => (
              <Link
                key={item.href}
                href={item.href}
                className="relative px-4 py-2 text-sm font-medium text-fg-secondary hover:text-fg-primary transition-colors rounded-lg hover:bg-bg-secondary"
              >
                <span>{item.label}</span>
                {item.badge && (
                  <Badge
                    variant="accent"
                    size="sm"
                    className="absolute -top-1 -right-1 animate-pulse"
                  >
                    {item.badge}
                  </Badge>
                )}
              </Link>
            ))}
          </div>

          {/* Right side */}
          <div className="flex items-center gap-4">
            {/* Connection status */}
            <div className="hidden sm:flex items-center gap-2">
              <div
                className={`w-2 h-2 rounded-full ${
                  isConnected ? 'bg-confidence-high animate-pulse' : 'bg-safety-danger'
                }`}
              />
              <span className="text-xs text-fg-tertiary">
                {isConnected ? 'Connected' : 'Disconnected'}
              </span>
              {health && (
                <SafetyBadge
                  level={
                    health.status === 'healthy'
                      ? 'safe'
                      : health.status === 'degraded'
                      ? 'caution'
                      : 'danger'
                  }
                  size="sm"
                  showLabel={false}
                />
              )}
            </div>

            {/* Theme toggle placeholder */}
            <Button variant="ghost" size="sm" className="hidden sm:flex">
              <ThemeIcon />
            </Button>

            {/* Settings */}
            <Link href="/settings">
              <Button variant="ghost" size="sm" className="hidden sm:flex">
                <SettingsIcon />
              </Button>
            </Link>

            {/* Mobile menu button */}
            <Button
              variant="ghost"
              size="sm"
              className="md:hidden"
              onClick={() => setMobileMenuOpen(!mobileMenuOpen)}
            >
              {mobileMenuOpen ? <CloseIcon /> : <MenuIcon />}
            </Button>
          </div>
        </div>

        {/* Mobile Navigation */}
        {mobileMenuOpen && (
          <div className="md:hidden border-t border-border-primary py-4">
            <div className="flex flex-col gap-2">
              {navItems.map((item) => (
                <Link
                  key={item.href}
                  href={item.href}
                  className="flex items-center justify-between px-4 py-3 text-fg-secondary hover:text-fg-primary hover:bg-bg-secondary rounded-lg transition-colors"
                  onClick={() => setMobileMenuOpen(false)}
                >
                  <span>{item.label}</span>
                  {item.badge && (
                    <Badge variant="accent" size="sm">
                      {item.badge}
                    </Badge>
                  )}
                </Link>
              ))}
            </div>

            {/* Settings link for mobile */}
            <Link
              href="/settings"
              className="flex items-center justify-between px-4 py-3 text-fg-secondary hover:text-fg-primary hover:bg-bg-secondary rounded-lg transition-colors"
              onClick={() => setMobileMenuOpen(false)}
            >
              <span>Settings</span>
            </Link>

            {/* Mobile connection status */}
            <div className="flex items-center gap-2 px-4 pt-4 mt-4 border-t border-border-primary">
              <div
                className={`w-2 h-2 rounded-full ${
                  isConnected ? 'bg-confidence-high animate-pulse' : 'bg-safety-danger'
                }`}
              />
              <span className="text-sm text-fg-tertiary">
                {isConnected ? 'Connected to HoloLoom' : 'Disconnected'}
              </span>
            </div>
          </div>
        )}
      </div>
    </nav>
  );
}

// Simple icon components
function ThemeIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2" />
      <path d="M12 20v2" />
      <path d="m4.93 4.93 1.41 1.41" />
      <path d="m17.66 17.66 1.41 1.41" />
      <path d="M2 12h2" />
      <path d="M20 12h2" />
      <path d="m6.34 17.66-1.41 1.41" />
      <path d="m19.07 4.93-1.41 1.41" />
    </svg>
  );
}

function SettingsIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="18"
      height="18"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M12.22 2h-.44a2 2 0 0 0-2 2v.18a2 2 0 0 1-1 1.73l-.43.25a2 2 0 0 1-2 0l-.15-.08a2 2 0 0 0-2.73.73l-.22.38a2 2 0 0 0 .73 2.73l.15.1a2 2 0 0 1 1 1.72v.51a2 2 0 0 1-1 1.74l-.15.09a2 2 0 0 0-.73 2.73l.22.38a2 2 0 0 0 2.73.73l.15-.08a2 2 0 0 1 2 0l.43.25a2 2 0 0 1 1 1.73V20a2 2 0 0 0 2 2h.44a2 2 0 0 0 2-2v-.18a2 2 0 0 1 1-1.73l.43-.25a2 2 0 0 1 2 0l.15.08a2 2 0 0 0 2.73-.73l.22-.39a2 2 0 0 0-.73-2.73l-.15-.08a2 2 0 0 1-1-1.74v-.5a2 2 0 0 1 1-1.74l.15-.09a2 2 0 0 0 .73-2.73l-.22-.38a2 2 0 0 0-2.73-.73l-.15.08a2 2 0 0 1-2 0l-.43-.25a2 2 0 0 1-1-1.73V4a2 2 0 0 0-2-2z" />
      <circle cx="12" cy="12" r="3" />
    </svg>
  );
}

function MenuIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <line x1="4" x2="20" y1="12" y2="12" />
      <line x1="4" x2="20" y1="6" y2="6" />
      <line x1="4" x2="20" y1="18" y2="18" />
    </svg>
  );
}

function CloseIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      width="20"
      height="20"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
    >
      <path d="M18 6 6 18" />
      <path d="m6 6 12 12" />
    </svg>
  );
}
