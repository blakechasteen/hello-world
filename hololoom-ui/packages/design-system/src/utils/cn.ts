/**
 * HoloLoom Design System - Class Name Utility
 *
 * Combines clsx and tailwind-merge for optimal class name handling.
 */

import { clsx, type ClassValue } from 'clsx';
import { twMerge } from 'tailwind-merge';

/**
 * Merge class names with Tailwind CSS conflict resolution.
 *
 * @example
 * cn('px-4 py-2', 'px-6') // → 'py-2 px-6'
 * cn('text-red-500', isActive && 'text-blue-500') // → 'text-blue-500' if active
 * cn({ 'bg-primary': isPrimary, 'bg-secondary': !isPrimary })
 */
export function cn(...inputs: ClassValue[]): string {
  return twMerge(clsx(inputs));
}
