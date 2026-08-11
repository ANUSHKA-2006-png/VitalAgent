interface Segment {
  value: number;
  color: string;
}

export default function DonutChart({
  segments,
  size = 120,
  thickness = 16,
}: {
  segments: Segment[];
  size?: number;
  thickness?: number;
}) {
  const radius = size / 2 - thickness / 2;
  const circumference = 2 * Math.PI * radius;
  const total = segments.reduce((sum, s) => sum + s.value, 0) || 1;

  let offset = 0;
  const arcs = segments.map((s, i) => {
    const fraction = s.value / total;
    const dash = fraction * circumference;
    const arc = (
      <circle
        key={i}
        r={radius}
        cx={size / 2}
        cy={size / 2}
        fill="transparent"
        stroke={s.color}
        strokeWidth={thickness}
        strokeDasharray={`${dash} ${circumference - dash}`}
        strokeDashoffset={-offset}
        transform={`rotate(-90 ${size / 2} ${size / 2})`}
      />
    );
    offset += dash;
    return arc;
  });

  return (
    <svg width={size} height={size} viewBox={`0 0 ${size} ${size}`}>
      {arcs}
    </svg>
  );
}
