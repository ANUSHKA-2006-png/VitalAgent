import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { Check, Loader2, Circle, HeartPulse } from "lucide-react";
import Header from "../components/Header";
import Stepper from "../components/Stepper";
import { analyzeScreening } from "../api/client";

export default function NewScreeningAnalysis() {
  const navigate = useNavigate();
  const [progress, setProgress] = useState(35);
  const [isDone, setIsDone] = useState(false);

  useEffect(() => {
    let isMounted = true;
    
    async function runAnalysis() {
      const detailsStr = sessionStorage.getItem("screening_details");
      const uploadId = sessionStorage.getItem("screening_upload_id") || undefined;
      const details = detailsStr ? JSON.parse(detailsStr) : { patientId: "P-1042" };

      if (isMounted) setProgress(55);

      const result = await analyzeScreening(
        details.patientId,
        uploadId,
        details.notes,
        {
          name: details.name,
          age: details.age,
          gender: details.gender,
          height: details.height,
          weight: details.weight,
          phone: details.phone,
        }
      );

      if (isMounted) {
        sessionStorage.setItem("screening_results", JSON.stringify(result));
        setProgress(100);
        setIsDone(true);
      }
    }

    const timer = setTimeout(() => {
      runAnalysis();
    }, 400);

    return () => {
      isMounted = false;
      clearTimeout(timer);
    };
  }, []);

  const steps = [
    { label: "Signal quality check (HR, Stress, SpO2, Motion)", status: "done" },
    { label: "Preparing 512-sample windows", status: progress >= 50 ? "done" : "active" },
    { label: "Running ILASH model", status: progress >= 80 ? "done" : progress >= 50 ? "active" : "pending" },
    { label: "Generating results", status: isDone ? "done" : progress >= 80 ? "active" : "pending" },
  ];

  return (
    <>
      <Header title="New Screening" backTo="/screening/new/upload" />
      <Stepper current={3} />
      <div className="flex-1 flex items-center justify-center p-6">
        <div className="grid grid-cols-1 sm:grid-cols-[220px_1fr] gap-8 max-w-3xl w-full items-center">
          <div className="h-44 rounded-card bg-gradient-to-br from-accent-light to-accent flex items-center justify-center">
            <HeartPulse size={56} className="text-white" />
          </div>

          <div>
            <div className="text-base font-medium mb-3">Analyzing Patient Data</div>
            <div className="space-y-2.5 mb-5">
              {steps.map((s) => (
                <div key={s.label} className="flex items-center gap-2 text-sm">
                  {s.status === "done" && <Check size={16} className="text-success-dot" />}
                  {s.status === "active" && (
                    <Loader2 size={16} className="text-accent animate-spin" />
                  )}
                  {s.status === "pending" && (
                    <Circle size={16} className="text-ink-muted" />
                  )}
                  <span className={s.status === "pending" ? "text-ink-muted" : ""}>
                    {s.label}
                  </span>
                  <span className="ml-auto text-xs text-ink-muted capitalize">
                    {s.status === "active" ? "In Progress" : s.status}
                  </span>
                </div>
              ))}
            </div>

            <div className="flex justify-between text-sm font-medium mb-1">
              <span>Overall Progress</span>
              <span>{progress}%</span>
            </div>
            <div className="h-2 rounded-full bg-wash overflow-hidden">
              <div
                className="h-full rounded-full bg-gradient-to-r from-accent-light to-accent transition-all duration-500"
                style={{ width: `${progress}%` }}
              />
            </div>
            <div className="text-[11px] text-ink-muted mt-2">
              {isDone ? "Analysis completed successfully!" : "Please wait while we analyze the data..."}
            </div>

            {progress >= 100 && (
              <button
                onClick={() => navigate("/screening/new/results")}
                className="mt-4 bg-gradient-to-r from-accent-light to-accent text-white text-sm font-medium rounded-control px-5 py-2.5"
              >
                View Results →
              </button>
            )}
          </div>
        </div>
      </div>
    </>
  );
}
