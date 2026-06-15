import { FILTER_OPTIONS } from "../utils/constants";

export default function FilterTabs({ activeFilter, onFilterChange, patients }) {
  return (
    <div style={{ maxWidth: 860, margin: "0 auto", padding: "20px 0 0" }}>
      <div style={{ display: "flex", gap: 6, flexWrap: "wrap" }}>
        {FILTER_OPTIONS.map(f => (
          <button
            key={f}
            onClick={() => onFilterChange(f)}
            style={{
              padding: "6px 16px",
              borderRadius: 999,
              border: activeFilter === f ? "none" : "1px solid #d1d5db",
              background: activeFilter === f ? "#111827" : "#fff",
              color: activeFilter === f ? "#fff" : "#6b7280",
              fontFamily: "'DM Mono', monospace",
              fontSize: 12, fontWeight: 600,
              cursor: "pointer",
              transition: "all 0.15s",
            }}
          >
            {f}
            {f !== "All" && (
              <span style={{ marginLeft: 6, opacity: 0.7 }}>
                {patients.filter(p => p.level === f).length}
              </span>
            )}
          </button>
        ))}
      </div>
    </div>
  );
}
