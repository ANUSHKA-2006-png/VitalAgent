import { useState, useEffect, ReactNode } from "react";
import { Link } from "react-router-dom";
import { ArrowLeft, Calendar, Clock } from "lucide-react";

export default function Header({
  title,
  subtitle,
  backTo,
  actions,
  showDate = true,
}: {
  title: string;
  subtitle?: string;
  backTo?: string;
  actions?: ReactNode;
  showDate?: boolean;
}) {
  const [now, setNow] = useState<Date>(new Date());

  useEffect(() => {
    const timer = setInterval(() => {
      setNow(new Date());
    }, 1000);
    return () => clearInterval(timer);
  }, []);

  const dateStr = now.toLocaleDateString("en-GB", {
    day: "2-digit",
    month: "short",
    year: "numeric",
  });

  const dayStr = now.toLocaleDateString("en-US", { weekday: "long" });
  
  const timeStr = now.toLocaleTimeString("en-US", {
    hour: "numeric",
    minute: "2-digit",
    second: "2-digit",
    hour12: true,
  });

  return (
    <div className="flex items-center justify-between h-16 px-6 border-b border-border bg-white flex-shrink-0">
      <div className="flex items-center gap-3 min-w-0">
        {backTo && (
          <Link
            to={backTo}
            className="text-ink-muted hover:text-ink flex-shrink-0"
          >
            <ArrowLeft size={20} />
          </Link>
        )}
        <div className="min-w-0">
          <div className="text-[17px] font-medium leading-tight truncate">
            {title}
          </div>
          {subtitle && (
            <div className="text-xs text-ink-muted leading-tight truncate">
              {subtitle}
            </div>
          )}
        </div>
      </div>
      <div className="flex items-center gap-3 flex-shrink-0">
        {actions}
        {showDate && (
          <div className="hidden lg:flex items-center gap-2 rounded-control bg-wash px-3 py-1.5 text-xs text-ink-muted">
            <Calendar size={14} className="text-accent" />
            <span className="font-medium text-ink">{dateStr}</span>
            <div className="text-[10px] text-ink-muted flex items-center gap-1 border-l border-border pl-2">
              <span>{dayStr}</span>
              <span>·</span>
              <span className="flex items-center gap-0.5">
                <Clock size={10} />
                {timeStr}
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
