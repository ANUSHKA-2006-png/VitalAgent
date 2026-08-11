const STEPS = ["Patient Details", "Data Upload", "Analysis", "Results"];

export default function Stepper({ current }: { current: 1 | 2 | 3 | 4 }) {
  return (
    <div className="flex items-center gap-3 px-6 py-4 text-xs text-ink-muted flex-wrap">
      {STEPS.map((step, i) => {
        const stepNum = i + 1;
        const active = stepNum === current;
        const done = stepNum < current;
        return (
          <div key={step} className="flex items-center gap-3">
            <div
              className={`flex items-center gap-2 ${
                active ? "text-accent font-medium" : ""
              }`}
            >
              <span
                className={`w-5 h-5 rounded-full flex items-center justify-center text-[11px] ${
                  active || done
                    ? "bg-accent text-white"
                    : "border border-border text-ink-muted"
                }`}
              >
                {stepNum}
              </span>
              {step}
            </div>
            {stepNum !== STEPS.length && <span>→</span>}
          </div>
        );
      })}
    </div>
  );
}
