import { getRiskLevel } from "../utils/scoring";
import { RISK_CONFIG } from "../utils/constants";

export default function ScoreBar({ score }) {
  const level = getRiskLevel(score);
  const cfg = RISK_CONFIG[level];

  return (
    <div style={{ display: "flex", alignItems: "center", gap: 8 }}>
      <div style={{
        flex: 1, height: 6, borderRadius: 3,
        background: "#e5e7eb", overflow: "hidden",
      }}>
        <div style={{
          width: `${score}%`, height: "100%",
          background: cfg.badge, borderRadius: 3,
          transition: "width 0.6s ease",
        }} />
      </div>
      <span style={{
        fontFamily: "'DM Mono', monospace",
        fontSize: 12, color: cfg.badge, fontWeight: 700,
        minWidth: 48, whiteSpace: "nowrap",
      }}>
        {score}<span style={{ opacity: 0.45, fontWeight: 400 }}>/100</span>
      </span>
    </div>
  );
}
