import { useState, useEffect } from "react";
import { useNavigate } from "react-router-dom";
import { Search, UserCheck } from "lucide-react";
import Header from "../components/Header";
import Stepper from "../components/Stepper";
import { fetchPatients } from "../api/client";
import { Patient } from "../types";

export default function NewScreeningDetails() {
  const navigate = useNavigate();

  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState<Patient[]>([]);
  const [showDropdown, setShowDropdown] = useState(false);

  const [patientId, setPatientId] = useState("P-1042");
  const [name, setName] = useState("Ramesh Kumar");
  const [age, setAge] = useState("54");
  const [gender, setGender] = useState("Male");
  const [height, setHeight] = useState("170");
  const [weight, setWeight] = useState("68");
  const [phone, setPhone] = useState("+91 98765 43210");
  const [notes, setNotes] = useState("");

  useEffect(() => {
    if (searchQuery.trim().length > 0) {
      fetchPatients(searchQuery).then((pts) => {
        setSearchResults(pts);
        setShowDropdown(true);
      });
    } else {
      setSearchResults([]);
      setShowDropdown(false);
    }
  }, [searchQuery]);

  const selectPatient = (p: Patient) => {
    setPatientId(p.id);
    setName(p.name);
    setAge(String(p.age));
    setGender(p.gender);
    setHeight(p.height ? String(p.height) : "170");
    setWeight(p.weight ? String(p.weight) : "68");
    setPhone(p.phone || "");
    setSearchQuery("");
    setShowDropdown(false);
  };

  const handleNext = () => {
    const details = {
      patientId: patientId || `P-${Math.floor(Math.random() * 9000 + 1000)}`,
      name,
      age,
      gender,
      height,
      weight,
      phone,
      notes,
    };
    sessionStorage.setItem("screening_details", JSON.stringify(details));
    navigate("/screening/new/upload");
  };

  return (
    <>
      <Header title="New Screening" backTo="/" />
      <Stepper current={1} />
      <div className="flex-1 overflow-y-auto p-6">
        <div className="grid grid-cols-1 lg:grid-cols-[1.5fr_1fr] gap-5 h-full">
          <div className="bg-white rounded-card border border-border p-6">
            <div className="flex items-center justify-between mb-4">
              <div className="text-sm font-medium">Patient Details</div>
              <div className="relative w-64">
                <Search size={14} className="absolute left-3 top-2.5 text-ink-muted" />
                <input
                  type="text"
                  placeholder="Search existing patient..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  onFocus={() => searchQuery.trim() && setShowDropdown(true)}
                  className="w-full h-8 rounded-control border border-border pl-8 pr-3 text-xs focus:outline-none focus:border-accent"
                />
                {showDropdown && searchResults.length > 0 && (
                  <div className="absolute top-9 left-0 right-0 bg-white border border-border rounded-control shadow-lg z-20 max-h-48 overflow-y-auto">
                    {searchResults.map((p) => (
                      <div
                        key={p.id}
                        onClick={() => selectPatient(p)}
                        className="px-3 py-2 text-xs hover:bg-wash cursor-pointer flex items-center justify-between"
                      >
                        <div>
                          <span className="font-medium text-ink">{p.name}</span>
                          <span className="text-ink-muted ml-1 font-mono">({p.id})</span>
                        </div>
                        <span className="text-[10px] text-accent flex items-center gap-0.5">
                          <UserCheck size={10} /> Select
                        </span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-4">
              <Field label="Patient ID" required value={patientId} onChange={(e) => setPatientId(e.target.value)} />
              <Field label="Age" required value={age} onChange={(e) => setAge(e.target.value)} />
            </div>

            <div className="mb-4">
              <Field label="Name" required value={name} onChange={(e) => setName(e.target.value)} />
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-3 gap-4 mb-4">
              <Field label="Gender" required value={gender} onChange={(e) => setGender(e.target.value)} />
              <Field label="Height (cm)" required value={height} onChange={(e) => setHeight(e.target.value)} />
              <Field label="Weight (kg)" required value={weight} onChange={(e) => setWeight(e.target.value)} />
            </div>

            <div className="mb-4">
              <Field label="Phone / Mobile" value={phone} onChange={(e) => setPhone(e.target.value)} />
            </div>

            <div>
              <label className="text-xs text-ink-muted block mb-1">Notes (optional)</label>
              <textarea
                value={notes}
                onChange={(e) => setNotes(e.target.value)}
                placeholder="Type any clinical notes..."
                className="w-full min-h-[75px] rounded-control border border-border px-3 py-2 text-sm focus:outline-none focus:border-accent"
              />
            </div>
          </div>

          <div className="bg-wash rounded-card p-6">
            <div className="text-sm font-medium mb-2">Selected Patient Summary</div>
            <div className="bg-white rounded-control border border-border p-4 mb-4 text-xs space-y-2">
              <div className="flex justify-between">
                <span className="text-ink-muted">Patient ID</span>
                <span className="font-mono font-medium">{patientId}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-ink-muted">Full Name</span>
                <span className="font-medium">{name}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-ink-muted">Age / Gender</span>
                <span>{age} Yrs · {gender}</span>
              </div>
              <div className="flex justify-between">
                <span className="text-ink-muted">Height / Weight</span>
                <span>{height} cm · {weight} kg</span>
              </div>
            </div>

            <div className="text-sm font-medium mb-2">About Screening</div>
            <p className="text-xs text-ink-muted leading-relaxed mb-4">
              VitalAgent AI analyzes continuous multi-modal physiological time-series signals to screen for cardiovascular health, stress, oxygen saturation, and fall risk.
            </p>
            <div className="text-sm font-medium mb-2">Supported Data Types</div>
            <ul className="text-xs text-ink-muted space-y-1.5 list-none">
              <li>✓ PPG + Accelerometer (Heart Rate)</li>
              <li>✓ WESAD Signals (Stress)</li>
              <li>✓ Pulse Oximeter Signal (SpO2)</li>
              <li>✓ Wrist Motion (Fall Detection)</li>
            </ul>
          </div>
        </div>

        <div className="flex justify-end mt-5">
          <button
            onClick={handleNext}
            className="bg-gradient-to-r from-accent-light to-accent text-white text-sm font-medium rounded-control px-5 py-2.5"
          >
            Next: Upload Data →
          </button>
        </div>
      </div>
    </>
  );
}

function Field({
  label,
  required,
  value,
  onChange,
}: {
  label: string;
  required?: boolean;
  value: string;
  onChange: (e: React.ChangeEvent<HTMLInputElement>) => void;
}) {
  return (
    <div>
      <label className="text-xs text-ink-muted block mb-1">
        {label} {required && <span className="text-danger-text">*</span>}
      </label>
      <input
        value={value}
        onChange={onChange}
        className="w-full h-9 rounded-control border border-border px-3 text-sm focus:outline-none focus:border-accent"
      />
    </div>
  );
}
