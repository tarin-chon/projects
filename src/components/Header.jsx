import { formatTodayLong } from "../utils/date";
import SummaryBar from "./SummaryBar";

export default function Header({ highRemaining, totalRemaining }) {
  return (
    <div style={{
      background: "#111827",
      padding: "28px 32px 24px",
      borderBottom: "1px solid #1f2937",
    }}>
      <div style={{ maxWidth: 860, margin: "0 auto" }}>
        <div style={{
          display: "flex", alignItems: "flex-start",
          justifyContent: "space-between", flexWrap: "wrap", gap: 16,
        }}>
          {/* Title block */}
          <div>
            <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4 }}>
              <div style={{
                width: 8, height: 8, borderRadius: "50%",
                background: "#ef4444",
                boxShadow: "0 0 0 3px #ef444433",
                animation: "pulse 2s infinite",
              }} />
              <span style={{ color: "#6b7280", fontSize: 11, letterSpacing: 2, textTransform: "uppercase" }}>
                Your Company Name Here · Care Coordination
              </span>
            </div>
            <h1 style={{
              margin: 0,
              fontFamily: "'Lora', serif",
              fontSize: 26, fontWeight: 700,
              color: "#f9fafb", letterSpacing: -0.3,
            }}>
              No-Show Risk Queue
            </h1>
            <p style={{ margin: "4px 0 0", color: "#9ca3af", fontSize: 12 }}>
              {formatTodayLong()}
            </p>
          </div>

          <SummaryBar highRemaining={highRemaining} totalRemaining={totalRemaining} />
        </div>
      </div>
    </div>
  );
}
