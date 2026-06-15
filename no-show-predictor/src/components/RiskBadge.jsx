import { RISK_CONFIG } from "../utils/constants";

export default function RiskBadge({ level }) {
  const cfg = RISK_CONFIG[level];

  return (
    <span style={{
      display: "inline-flex", alignItems: "center", gap: 4,
      padding: "2px 10px", borderRadius: 999,
      background: cfg.bg, border: `1px solid ${cfg.border}`,
      color: cfg.badge, fontSize: 11, fontWeight: 700,
      fontFamily: "'DM Mono', monospace", letterSpacing: 0.5,
    }}>
      <span style={{
        width: 6, height: 6, borderRadius: "50%",
        background: cfg.badge, display: "inline-block",
      }} />
      {level.toUpperCase()}
    </span>
  );
}
