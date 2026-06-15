import PatientCard from "./PatientCard";
import ContactedRow from "./ContactedRow";

export default function PatientList({ filtered, contactedPatients, onContact, onUndo }) {
  return (
    <div style={{ maxWidth: 860, margin: "16px auto 0", display: "flex", flexDirection: "column", gap: 10 }}>

      {/* Active patients */}
      {filtered.length === 0 ? (
        <div style={{
          textAlign: "center", padding: "48px 0",
          color: "#9ca3af", fontFamily: "'DM Mono', monospace", fontSize: 13,
        }}>
          No patients in this category.
        </div>
      ) : (
        filtered.map(patient => (
          <PatientCard key={patient.id} patient={patient} onContact={onContact} />
        ))
      )}

      {/* Contacted this shift */}
      {contactedPatients.length > 0 && (
        <div style={{ marginTop: 24 }}>
          <div style={{
            fontSize: 10, letterSpacing: 2, color: "#9ca3af",
            textTransform: "uppercase", marginBottom: 8,
            borderTop: "1px solid #e5e7eb", paddingTop: 20,
          }}>
            Contacted this shift ({contactedPatients.length})
          </div>
          <div style={{ display: "flex", flexDirection: "column", gap: 6 }}>
            {contactedPatients.map(p => (
              <ContactedRow key={p.id} patient={p} onUndo={onUndo} />
            ))}
          </div>
        </div>
      )}

    </div>
  );
}
