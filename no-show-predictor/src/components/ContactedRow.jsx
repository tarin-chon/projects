export default function ContactedRow({ patient, onUndo }) {
  return (
    <div style={{
      display: "flex", alignItems: "center", justifyContent: "space-between",
      padding: "10px 16px",
      background: "#f9fafb", border: "1px solid #e5e7eb",
      borderRadius: 8, opacity: 0.75,
    }}>
      <div style={{ display: "flex", alignItems: "center", gap: 10 }}>
        <span style={{ fontSize: 14, color: "#16a34a" }}>✓</span>
        <span style={{ fontFamily: "'Lora', serif", fontSize: 14, color: "#6b7280", textDecoration: "line-through" }}>
          {patient.name}
        </span>
        <span style={{ fontFamily: "'DM Mono', monospace", fontSize: 11, color: "#9ca3af" }}>
          contacted
        </span>
      </div>
      <button
        onClick={() => onUndo(patient.id)}
        style={{
          background: "none", border: "none", color: "#9ca3af",
          fontFamily: "'DM Mono', monospace", fontSize: 11,
          cursor: "pointer", textDecoration: "underline", padding: 0,
        }}
      >
        undo
      </button>
    </div>
  );
}
