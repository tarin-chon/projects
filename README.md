# No-Show Risk Queue

A care coordination tool that scores and ranks patients by appointment no-show risk, so coordinators spend their limited call time on the patients who need it most. 

---

## 1. Approach

The core insight behind this tool is that a care coordinator making 15–20 calls per shift doesn't need a list — they need a ranked queue. The UI is built around that workflow: highest-risk patients appear first, each card gives enough context to act immediately, and marking a patient as contacted removes them from the active queue without losing the record of who was reached.

---

## 2. Scoring Logic

Each patient receives a risk score from 0–100, computed from the following factors in descending order of weight:

**No-show rate — 40 pts**
The ratio of missed appointments to total appointments (`priorNoShows / priorAppts`), rather than the raw count. A patient who has missed 3 of 4 appointments (75%) is meaningfully higher risk than one who has missed 3 of 20 (15%). New patients with no prior visit history receive 0 points here (handled separately below).

**Socioeconomic barriers — 15 pts**
Proxied through insurance type. Medi-Cal patients face greater transportation barriers and are more likely to have inflexible work schedules, making it harder to attend appointments. Medi-Cal receives full weight (1.0×), Medicare Advantage receives partial weight (0.5×), and Commercial receives none.

**New patient — 15 pts**
Inferred from `priorAppts === 0`. New patients have no established relationship with the clinic and no prior visit pattern, which correlates with higher no-show rates. This is treated as a flat bonus rather than a rate calculation since there is no history to compute a rate from.

**Distance — 15 pts**
Scored linearly up to a 25-mile cap. Farther patients face greater logistical challenges. Distance is shown in the card explanation only when it exceeds 15 miles, since shorter distances are unlikely to be a meaningful factor.

**Appointment urgency — 15 pts**
Appointments sooner get more points, scaled from 0 to 30 days out. This reflects coordinator priority rather than no-show probability — the window to intervene closes as the appointment approaches, so a high-risk patient with an appointment tomorrow is more urgent than the same patient booked three weeks from now.

**Thresholds:**
- High: 67–100
- Medium: 34–66
- Low: 0–33

---

## 3. Tradeoffs

**No-show rate over raw count.** A raw count of prior no-shows is misleading without knowing how many total appointments a patient has had. 

**Recency weighting is noted but not implemented.** Recent no-shows are more predictive than older ones. For example, a patient who missed a couple appointments in the past month is a stronger signal than a cluster of no-shows 5 years ago. The sample data doesn't include per-appointment timestamps, so recency weighting was omitted. 

**Disability status is noted but not in the sample data.** Patients with physical disabilities or mobility limitations face significant transportation barriers and are a strong predictor of no-show risk. This would warrant its own scoring factor given patient records that include it.

**Session-only contacted state.** Marking a patient as contacted is not persisted across page refreshes. In a real deployment, this state would live server-side and be scoped to a shift, so multiple coordinators share a single source of truth and the next shift can see who was already reached. This was omitted intentionally given the frontend-only scope of the exercise.

**Insurance as a socioeconomic proxy.** Insurance type is a rough signal — not all Medi-Cal patients face transportation barriers, and not all Commercial patients lack them. In production, more direct indicators (documented transportation barriers, prior cancellation patterns, distance combined with transit access) would replace or supplement this.

---

## 4. UX Decisions

**Ranked by score, not name or date.** The list is sorted highest-to-lowest risk so coordinators can start at the top and work down without needing to interpret the data themselves.

**Plain-language explanation on every card.** Rather than showing raw numbers, each card surfaces a brief, readable summary. The goal is that a coordinator should be able to read a card and immediately understand why this patient is flagged.

**Filter tabs by risk level.** Coordinators can narrow to High-risk only when time is short, or scan Medium-risk patients later in the shift. The count per tier updates as patients are contacted.

**"Mark Contacted" with undo.** Contacted patients move to a struck-through log at the bottom of the queue rather than disappearing entirely, with an undo affordance in case of accidental taps. 

**Patient avatars as a humanization layer.** Each patient card includes an avatar component that currently renders a colored circle with the patient's initials. The component is built to accept a `photoUrl` prop — when a real patient photo is available, passing it in swaps the initials for the photo with no changes needed elsewhere in the codebase. The intent is that seeing a patient's face, rather than a name on a list, creates a stronger human connection for the coordinator before they pick up the phone. A coordinator who feels that connection is more likely to approach the call with empathy and provide more tailored, attentive service.

**Summary chips in the header.** High-risk remaining and total remaining are surfaced at a glance. When all high-risk patients are contacted, the respective chip turns green, giving coordinators a clear milestone signal. The same is true for the total remaining summary chip.

---

## 5. What I'd Do Next

**Multi-shift support.** The current tool is scoped to a single session. With a backend, contacted state would be persisted and scoped to a shift boundary, with a handoff log so the incoming coordinator knows who was already reached. A manual "End Shift" action would close the queue and archive the session.

**Recency-weighted no-show scoring.** With timestamped appointment history, recent no-shows would carry more weight than older no-shows. 

**Disability and transportation flags.** Direct patient-level indicators (e.g. documented mobility limitations, no car on file) would sharpen the socioeconomic signal beyond what insurance type can proxy.

**Coordinator notes per patient.** A lightweight free-text field on each card — "left voicemail," "wrong number," "patient confirmed" — would make the contacted log actionable rather than just a count.

**Outcome tracking.** Logging whether contacted patients actually showed up would let the scoring model be validated and improved over time. Over enough cycles, actual no-show outcomes could be used to recalibrate the factor weights.

**Real data integration.** `src/data/patients.js` is intentionally isolated: Replacing the mock data with a real database query is a one-file change. The scoring logic in `src/utils/scoring.js` has no UI dependencies and could be unit tested independently before connecting to real data.

---

## 6. Deployment

The app is live at: **[https://appointmentno-showpredictor-virid.vercel.app/]**

Built with React and deployed on Vercel.

### Run locally

If the live link is unavailable, you can run the app locally in a few steps:

```bash
# 1. Clone the repository
git clone https://github.com/tarin-chon/appointment_no-show_predictor.git
cd appointment_no-show_predictor

# 2. Install dependencies
npm install

# 3. Start the development server
npm run dev
```

Then open [http://localhost:5173](http://localhost:5173) in your browser (or whichever port Vite assigns).

Node 18+ is recommended.
