import { AVATAR_COLORS } from "../utils/constants";

// Accepts an optional `photoUrl` prop.
// When a real patient photo is available, pass it in and the image will render
// in place of the initials fallback — no other changes needed at the call site.
//
// Example with photo:   <Avatar name="Maria Garcia" id={1} photoUrl={patient.photoUrl} />
// Example without photo: <Avatar name="Maria Garcia" id={1} />
export default function Avatar({ name, id, photoUrl }) {
  const initials = name.split(" ").map(n => n[0]).join("").slice(0, 2);
  const color = AVATAR_COLORS[id % AVATAR_COLORS.length];

  const sharedStyles = {
    width: 40, height: 40, borderRadius: "50%",
    flexShrink: 0, overflow: "hidden",
  };

  if (photoUrl) {
    return (
      <img
        src={photoUrl}
        alt={name}
        style={{ ...sharedStyles, objectFit: "cover" }}
      />
    );
  }

  return (
    <div style={{
      ...sharedStyles,
      background: color, color: "#fff",
      display: "flex", alignItems: "center", justifyContent: "center",
      fontFamily: "'DM Mono', monospace", fontSize: 13, fontWeight: 600,
      letterSpacing: 1,
    }}>
      {initials}
    </div>
  );
}
