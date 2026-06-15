export default function SummaryBar({ highRemaining, totalRemaining }) {
  return (
    <div style={{ display: "flex", gap: 10, flexWrap: "wrap" }}>

      {/* High risk chip */}
      {highRemaining > 0 ? (
        <div style={{
          background: "#fef2f2", border: "1px solid #fecaca",
          borderRadius: 8, padding: "10px 18px", textAlign: "center",
        }}>
          <div style={{ fontSize: 22, fontWeight: 700, color: "#dc2626", fontFamily: "'Lora', serif" }}>
            {highRemaining}
          </div>
          <div style={{ fontSize: 10, color: "#dc2626", letterSpacing: 1, textTransform: "uppercase" }}>
            High Risk
          </div>
        </div>
      ) : (
        <div style={{
          background: "#f0fdf4", border: "1px solid #bbf7d0",
          borderRadius: 8, padding: "10px 18px", textAlign: "center",
        }}>
          <div style={{ fontSize: 14, color: "#16a34a", fontWeight: 600 }}>✓ All high-risk</div>
          <div style={{ fontSize: 10, color: "#16a34a", letterSpacing: 1 }}>contacted</div>
        </div>
      )}

      {/* Total remaining chip */}
      {totalRemaining > 0 ? (
        <div style={{
          background: "#1f2937", border: "1px solid #374151",
          borderRadius: 8, padding: "10px 18px", textAlign: "center",
        }}>
          <div style={{ fontSize: 22, fontWeight: 700, color: "#f9fafb", fontFamily: "'Lora', serif" }}>
            {totalRemaining}
          </div>
          <div style={{ fontSize: 10, color: "#9ca3af", letterSpacing: 1, textTransform: "uppercase" }}>
            Total Remaining
          </div>
        </div>
      ) : (
        <div style={{
          background: "#f0fdf4", border: "1px solid #bbf7d0",
          borderRadius: 8, padding: "10px 18px", textAlign: "center",
        }}>
          <div style={{ fontSize: 14, color: "#16a34a", fontWeight: 600 }}>✓ All patients</div>
          <div style={{ fontSize: 10, color: "#16a34a", letterSpacing: 1 }}>contacted</div>
        </div>
      )}

    </div>
  );
}
