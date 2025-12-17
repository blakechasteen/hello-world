'use client';

interface ConfidenceIndicatorProps {
  confidence: number;
  size?: 'sm' | 'md' | 'lg';
  showLabel?: boolean;
  showBar?: boolean;
  animated?: boolean;
}

export function ConfidenceIndicator({
  confidence,
  size = 'md',
  showLabel = true,
  showBar = true,
  animated = true,
}: ConfidenceIndicatorProps) {
  const percentage = Math.round(confidence * 100);

  const sizeClasses = {
    sm: {
      text: 'text-xs',
      bar: 'h-1 w-12',
      ring: 'w-8 h-8',
    },
    md: {
      text: 'text-sm',
      bar: 'h-1.5 w-16',
      ring: 'w-12 h-12',
    },
    lg: {
      text: 'text-base',
      bar: 'h-2 w-24',
      ring: 'w-16 h-16',
    },
  };

  const getColor = (value: number) => {
    if (value >= 0.85) return { bg: 'bg-safety-safe', text: 'text-safety-safe', stroke: '#10B981' };
    if (value >= 0.7) return { bg: 'bg-safety-safe', text: 'text-safety-safe', stroke: '#10B981' };
    if (value >= 0.5) return { bg: 'bg-safety-risky', text: 'text-safety-risky', stroke: '#F59E0B' };
    return { bg: 'bg-safety-critical', text: 'text-safety-critical', stroke: '#EF4444' };
  };

  const colors = getColor(confidence);

  const getLabel = (value: number): string => {
    if (value >= 0.85) return 'High';
    if (value >= 0.7) return 'Good';
    if (value >= 0.5) return 'Moderate';
    return 'Low';
  };

  return (
    <div className="flex items-center gap-2">
      {showBar && (
        <div className={`${sizeClasses[size].bar} bg-bg-tertiary rounded-full overflow-hidden`}>
          <div
            className={`h-full rounded-full ${colors.bg} ${animated ? 'transition-all duration-500' : ''}`}
            style={{ width: `${percentage}%` }}
          />
        </div>
      )}

      <span className={`font-medium ${sizeClasses[size].text} ${colors.text}`}>
        {percentage}%
      </span>

      {showLabel && (
        <span className={`${sizeClasses[size].text} text-fg-tertiary`}>
          ({getLabel(confidence)})
        </span>
      )}
    </div>
  );
}

interface ConfidenceRingProps {
  confidence: number;
  size?: 'sm' | 'md' | 'lg';
  showPercentage?: boolean;
}

export function ConfidenceRing({
  confidence,
  size = 'md',
  showPercentage = true,
}: ConfidenceRingProps) {
  const percentage = Math.round(confidence * 100);

  const sizes = {
    sm: { width: 40, strokeWidth: 3, fontSize: 10 },
    md: { width: 60, strokeWidth: 4, fontSize: 14 },
    lg: { width: 80, strokeWidth: 5, fontSize: 18 },
  };

  const { width, strokeWidth, fontSize } = sizes[size];
  const radius = (width - strokeWidth) / 2;
  const circumference = 2 * Math.PI * radius;
  const strokeDashoffset = circumference - confidence * circumference;

  const getColor = (value: number): string => {
    if (value >= 0.7) return '#10B981';
    if (value >= 0.5) return '#F59E0B';
    return '#EF4444';
  };

  return (
    <svg width={width} height={width} viewBox={`0 0 ${width} ${width}`}>
      {/* Background circle */}
      <circle
        cx={width / 2}
        cy={width / 2}
        r={radius}
        fill="none"
        stroke="currentColor"
        strokeOpacity="0.1"
        strokeWidth={strokeWidth}
      />

      {/* Progress circle */}
      <circle
        cx={width / 2}
        cy={width / 2}
        r={radius}
        fill="none"
        stroke={getColor(confidence)}
        strokeWidth={strokeWidth}
        strokeLinecap="round"
        strokeDasharray={circumference}
        strokeDashoffset={strokeDashoffset}
        transform={`rotate(-90 ${width / 2} ${width / 2})`}
        style={{ transition: 'stroke-dashoffset 0.5s ease' }}
      />

      {/* Percentage text */}
      {showPercentage && (
        <text
          x={width / 2}
          y={width / 2}
          textAnchor="middle"
          dominantBaseline="central"
          className="fill-fg-primary font-semibold"
          style={{ fontSize }}
        >
          {percentage}%
        </text>
      )}
    </svg>
  );
}

interface ConfidenceBadgeProps {
  confidence: number;
}

export function ConfidenceBadge({ confidence }: ConfidenceBadgeProps) {
  const percentage = Math.round(confidence * 100);

  const getVariant = (value: number) => {
    if (value >= 0.7) return 'bg-safety-safe/10 text-safety-safe border-safety-safe/20';
    if (value >= 0.5) return 'bg-safety-risky/10 text-safety-risky border-safety-risky/20';
    return 'bg-safety-critical/10 text-safety-critical border-safety-critical/20';
  };

  const getLabel = (value: number): string => {
    if (value >= 0.85) return 'High Confidence';
    if (value >= 0.7) return 'Good Confidence';
    if (value >= 0.5) return 'Moderate Confidence';
    return 'Low Confidence';
  };

  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2 py-1 rounded-full text-xs font-medium border ${getVariant(confidence)}`}
    >
      <span className="w-1.5 h-1.5 rounded-full bg-current" />
      {percentage}% · {getLabel(confidence)}
    </span>
  );
}
