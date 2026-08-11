import { Routes, Route } from "react-router-dom";
import Layout from "./components/Layout";
import Dashboard from "./pages/Dashboard";
import NewScreeningDetails from "./pages/NewScreeningDetails";
import NewScreeningUpload from "./pages/NewScreeningUpload";
import NewScreeningAnalysis from "./pages/NewScreeningAnalysis";
import NewScreeningResults from "./pages/NewScreeningResults";
import Patients from "./pages/Patients";
import PatientProfile from "./pages/PatientProfile";
import Alerts from "./pages/Alerts";
import Reports from "./pages/Reports";
import CommunityAnalytics from "./pages/CommunityAnalytics";
import Settings from "./pages/Settings";

export default function App() {
  return (
    <Routes>
      <Route element={<Layout />}>
        <Route path="/" element={<Dashboard />} />
        <Route path="/screening/new/details" element={<NewScreeningDetails />} />
        <Route path="/screening/new/upload" element={<NewScreeningUpload />} />
        <Route path="/screening/new/analysis" element={<NewScreeningAnalysis />} />
        <Route path="/screening/new/results" element={<NewScreeningResults />} />
        <Route path="/patients" element={<Patients />} />
        <Route path="/patients/:id" element={<PatientProfile />} />
        <Route path="/alerts" element={<Alerts />} />
        <Route path="/reports" element={<Reports />} />
        <Route path="/analytics" element={<CommunityAnalytics />} />
        <Route path="/settings" element={<Settings />} />
      </Route>
    </Routes>
  );
}
