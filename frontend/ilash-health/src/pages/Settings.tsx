import Header from "../components/Header";

export default function Settings() {
  return (
    <>
      <Header title="Settings" subtitle="Manage your account and preferences." />
      <div className="flex-1 overflow-y-auto p-6">
        <div className="bg-white rounded-card border border-border p-8 text-center text-sm text-ink-muted">
          Settings screen — add account, notification, and sensor-calibration
          preferences here.
        </div>
      </div>
    </>
  );
}
