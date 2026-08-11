import { Outlet } from "react-router-dom";
import Sidebar from "./Sidebar";

export default function Layout() {
  return (
    <div className="flex h-screen w-screen overflow-hidden bg-wash">
      <Sidebar />
      <div className="flex-1 flex flex-col min-w-0 h-full">
        <Outlet />
      </div>
    </div>
  );
}
