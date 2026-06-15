import { useState, useMemo } from "react";
import PATIENTS from "./data/patients";
import { scoreAndSortPatients } from "./utils/scoring";
import Header from "./components/Header";
import FilterTabs from "./components/FilterTabs";
import PatientList from "./components/PatientList";

export default function App() {
  const [contacted, setContacted] = useState(new Set());
  const [filter, setFilter] = useState("All");

  // Score and sort once on mount
  const scored = useMemo(() => scoreAndSortPatients(PATIENTS), []);

  const active = scored.filter(p => !contacted.has(p.id));
  const contactedPatients = scored.filter(p => contacted.has(p.id));
  const filtered = filter === "All" ? active : active.filter(p => p.level === filter);

  const highRemaining = active.filter(p => p.level === "High").length;

  const handleContact = (id) =>
    setContacted(prev => new Set([...prev, id]));

  const handleUndo = (id) =>
    setContacted(prev => { const s = new Set(prev); s.delete(id); return s; });

  return (
    <div style={{ minHeight: "100vh", background: "#f8f7f4", fontFamily: "'DM Mono', monospace", padding: "0 0 60px" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=Lora:wght@400;600;700&family=DM+Mono:wght@400;500;600&display=swap');
        * { box-sizing: border-box; }
        body { margin: 0; }
        @keyframes pulse {
          0%, 100% { opacity: 1; }
          50% { opacity: 0.4; }
        }
      `}</style>

      <Header highRemaining={highRemaining} totalRemaining={active.length} />

      <FilterTabs
        activeFilter={filter}
        onFilterChange={setFilter}
        patients={active}
      />

      <PatientList
        filtered={filtered}
        contactedPatients={contactedPatients}
        onContact={handleContact}
        onUndo={handleUndo}
      />
    </div>
  );
}
