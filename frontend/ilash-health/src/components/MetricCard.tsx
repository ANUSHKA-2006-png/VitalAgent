import { ReactNode } from "react";

export default function MetricCard({
  label,
  value,
  trend,
  trendUp = true,
  icon,
}: {
  label: string;
  value: string;
  trend?: string;
  trendUp?: boolean;
  icon?: ReactNode;
}) {
  return (
    <div className="bg-wash rounded-card p-4 flex-1 min-w-0">
      <div className="flex items-center justify-between text-ink-muted text-xs mb-1">
        {label}
        {icon}
      </div>
      <div className="text-2xl font-semibold leading-tight">{value}</div>
      {trend && (
        <div
          className={`text-[11px] mt-1 ${
            trendUp ? "text-success-text" : "text-danger-text"
          }`}
        >
          {trendUp ? "↑" : "↓"} {trend}
        </div>
      )}
    </div>
  );
}
