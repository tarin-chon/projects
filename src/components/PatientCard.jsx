import { buildExplanation } from "../utils/scoring";
import { formatApptDate } from "../utils/date";
import { RISK_CONFIG } from "../utils/constants";
import Avatar from "./Avatar";
import RiskBadge from "./RiskBadge";
import ScoreBar from "./ScoreBar";

export default function PatientCard({ patient, onContact }) {
  const { score, level } = patient;
  const cfg = RISK_CONFIG[level];
  const explanation = buildExplanation(patient);

  return (
    <div style={{
      background: "#fff",
      border: `1px solid ${level === "High" ? cfg.border : "#e5e7eb"}`,
      borderLeft: `4px solid ${cfg.badge}`,
      borderRadius: 10,
      padding: "16px 20px",
      display: "flex", alignItems: "center", gap: 16,
      boxShadow: level === "High" ? `0 2px 12px ${cfg.color}22` : "0 1px 4px rgba(0,0,0,0.06)",
      transition: "box-shadow 0.2s",
    }}>
      <Avatar name={patient.name} id={patient.id} />

      <div style={{ flex: 1, minWidth: 0 }}>
        {/* Name, badge, insurance */}
        <div style={{ display: "flex", alignItems: "center", gap: 10, marginBottom: 4, flexWrap: "wrap" }}>
          <span style={{ fontFamily: "'Lora', serif", fontSize: 16, fontWeight: 600, color: "#111827" }}>
            {patient.name}
          </span>
          <RiskBadge level={level} />
          <span style={{
            fontSize: 11, color: "#6b7280",
            fontFamily: "'DM Mono', monospace",
            background: "#f3f4f6", borderRadius: 4, padding: "1px 6px",
          }}>
            {patient.insurance}
          </span>
        </div>

        {/* Score bar */}
        <div style={{ marginBottom: 8 }}>
          <ScoreBar score={score} />
        </div>

        {/* Plain-language explanation */}
        <div style={{ fontSize: 12, color: "#6b7280", fontFamily: "'DM Mono', monospace", lineHeight: 1.5 }}>
          {explanation}
        </div>

        {/* Appointment date */}
        <div style={{ marginTop: 6, fontSize: 12, color: "#9ca3af", fontFamily: "'DM Mono', monospace" }}>
          Appt: {formatApptDate(patient.apptDate)}
        </div>
      </div>

      <button
        onClick={() => onContact(patient.id)}
        style={{
          padding: "8px 18px",
          background: "#111827", color: "#fff",
          border: "none", borderRadius: 7,
          fontFamily: "'DM Mono', monospace", fontSize: 12,
          fontWeight: 600, cursor: "pointer",
          whiteSpace: "nowrap", letterSpacing: 0.3,
          transition: "background 0.15s",
          flexShrink: 0,
        }}
        onMouseEnter={e => e.target.style.background = "#374151"}
        onMouseLeave={e => e.target.style.background = "#111827"}
      >
        Mark Contacted
      </button>
    </div>
  );
}
