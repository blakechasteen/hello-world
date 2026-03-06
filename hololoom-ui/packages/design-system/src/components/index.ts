/**
 * HoloLoom Design System - Component Exports
 *
 * All primitive and HoloLoom-specific UI components.
 */

// Primitives
export { Button, type ButtonProps, type ButtonVariant, type ButtonSize } from './Button';
export { Input, Textarea, type InputProps, type TextareaProps, type InputVariant, type InputSize } from './Input';
export {
  Card,
  CardHeader,
  CardBody,
  CardFooter,
  type CardProps,
  type CardHeaderProps,
  type CardVariant,
} from './Card';

// HoloLoom-specific
export {
  ConfidenceIndicator,
  getConfidenceLevel,
  getConfidenceLabel,
  type ConfidenceIndicatorProps,
  type ConfidenceLevel,
  type ConfidenceVariant,
  type ConfidenceSize,
} from './ConfidenceIndicator';

export {
  SafetyBadge,
  getSafetyLabel,
  type SafetyBadgeProps,
  type SafetyLevel,
  type SafetySize,
} from './SafetyBadge';

export {
  MemoryNode,
  type MemoryNodeProps,
  type MemoryState,
  type MemoryNodeSize,
} from './MemoryNode';

export {
  ThreadItem,
  type ThreadItemProps,
  type ThreadStatus,
} from './ThreadItem';

export {
  ReasoningBadge,
  type ReasoningBadgeProps,
  type ReasoningBadgeSize,
  type ReasoningMode,
} from './ReasoningBadge';

export {
  LoadingSpinner,
  LoadingDots,
  StreamingCursor,
  WeavingIndicator,
  type LoadingSpinnerProps,
  type LoadingDotsProps,
  type StreamingCursorProps,
  type WeavingIndicatorProps,
  type SpinnerSize,
} from './Loading';

export {
  Badge,
  type BadgeProps,
  type BadgeVariant,
} from './Badge';

export {
  Avatar,
  type AvatarProps,
  type AvatarSize,
  type AvatarVariant,
} from './Avatar';

export {
  Tooltip,
  InfoTooltip,
  type TooltipProps,
  type InfoTooltipProps,
  type TooltipPosition,
} from './Tooltip';

export {
  AvatarGroup,
  type AvatarGroupProps,
} from './Avatar';
