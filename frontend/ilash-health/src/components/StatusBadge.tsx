import { Status } from "../types";

const STYLES: Record<Status, string> = {
  normal: "bg-success-bg text-success-text",
  attention: "bg-attention-bg text-attention-text",
  high: "bg-danger-bg text-danger-text",
};

const LABELS: Record<Status, string> = {
  normal: "Normal",
  attention: "Attention",
  high: "High Priority",
};

export default function StatusBadge({
  status,
  label,
}: {
  status: Status;
  label?: string;
}) {
  return (
    <span
      className={`inline-flex items-center rounded-control px-2 py-0.5 text-[11px] font-medium ${STYLES[status]}`}
    >
      {label ?? LABELS[status]}
    </span>
  );
}
