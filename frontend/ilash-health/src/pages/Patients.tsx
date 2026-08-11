import { useState, useEffect } from "react";
import { Link } from "react-router-dom";
import { Search, SlidersHorizontal, Plus, Eye, ChevronLeft, ChevronRight } from "lucide-react";
import Header from "../components/Header";
import StatusBadge from "../components/StatusBadge";
import { fetchPatients } from "../api/client";
import { Patient, Status } from "../types";

export default function Patients() {
  const [query, setQuery] = useState("");
  const [patientList, setPatientList] = useState<Patient[]>([]);

  useEffect(() => {
    fetchPatients(query).then(setPatientList);
  }, [query]);

  return (
    <>
      <Header
        title="Patients"
        subtitle="Manage and view all registered patients."
        actions={
          <>
            <div className="hidden md:flex items-center gap-2 border border-border rounded-control px-3 py-2 text-xs w-52">
              <Search size={14} className="text-ink-muted" />
              <input
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Search patients..."
                className="flex-1 outline-none text-xs"
              />
            </div>
            <button className="hidden sm:flex items-center gap-1.5 border border-border rounded-control px-3 py-2 text-xs bg-white">
              <SlidersHorizontal size={14} /> Filters
            </button>
            <Link
              to="/screening/new/details"
              className="flex items-center gap-1.5 bg-gradient-to-r from-accent-light to-accent text-white text-sm font-medium rounded-control px-4 py-2"
            >
              <Plus size={16} /> Add New Patient
            </Link>
          </>
        }
      />

      <div className="flex-1 overflow-y-auto p-6">
        <div className="bg-white rounded-card border border-border overflow-hidden">
          <table className="w-full text-sm">
            <thead>
              <tr className="bg-wash text-ink-muted text-left text-xs">
                <th className="font-normal px-4 py-3">Patient ID</th>
                <th className="font-normal px-4 py-3">Name</th>
                <th className="font-normal px-4 py-3">Gender</th>
                <th className="font-normal px-4 py-3">SpO2</th>
                <th className="font-normal px-4 py-3">Last Screening</th>
                <th className="font-normal px-4 py-3">Status</th>
                <th className="font-normal px-4 py-3">Actions</th>
              </tr>
            </thead>
            <tbody>
              {patientList.map((p) => (
                <tr key={p.id} className="border-t border-border hover:bg-wash">
                  <td className="px-4 py-3">{p.id}</td>
                  <td className="px-4 py-3">{p.name}</td>
                  <td className="px-4 py-3">{p.gender}</td>
                  <td className="px-4 py-3">{p.spo2}%</td>
                  <td className="px-4 py-3">{p.lastScreening}</td>
                  <td className="px-4 py-3">
                    <StatusBadge status={p.status as Status} />
                  </td>
                  <td className="px-4 py-3">
                    <Link to={`/patients/${p.id}`} className="text-ink-muted">
                      <Eye size={16} />
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="flex items-center justify-between mt-3 text-xs text-ink-muted">
          <div>Showing 1 to {patientList.length} of {patientList.length} patients</div>
          <div className="flex items-center gap-1">
            <button className="p-1.5 border border-border rounded-control">
              <ChevronLeft size={14} />
            </button>
            <button className="w-7 h-7 rounded-control text-xs bg-accent text-white">1</button>
            <button className="p-1.5 border border-border rounded-control">
              <ChevronRight size={14} />
            </button>
          </div>
        </div>
      </div>
    </>
  );
}
