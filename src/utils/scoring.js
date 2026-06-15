import { daysUntil } from "./date";

// Risk level thresholds
const THRESHOLDS = { HIGH: 67, MEDIUM: 34 };

// Insurance socioeconomic weights (0–1)
const INSURANCE_WEIGHTS = {
  "Medi-Cal": 1.0,
  "Medicare Advantage": 0.5,
  "Commercial": 0.0,
};

// Point allocations per factor (must sum to 100)
const WEIGHTS = {
  noShowRate: 40,  // strongest predictor
  socioeconomic: 15,
  newPatient: 15,
  distance: 15,
  urgency: 15,
};

const MAX_DISTANCE_MILES = 25;
const MAX_URGENCY_DAYS = 30;

function getNoShowRatePoints(priorNoShows, priorAppts) {
  if (priorAppts === 0) return 0;
  return (priorNoShows / priorAppts) * WEIGHTS.noShowRate;
}

function getSocioeconomicPoints(insurance) {
  const weight = INSURANCE_WEIGHTS[insurance] ?? 0;
  return weight * WEIGHTS.socioeconomic;
}

function getNewPatientPoints(priorAppts) {
  return priorAppts === 0 ? WEIGHTS.newPatient : 0;
}

function getDistancePoints(distanceMiles) {
  return Math.min(distanceMiles / MAX_DISTANCE_MILES, 1) * WEIGHTS.distance;
}

function getUrgencyPoints(days) {
  // Appointments sooner = higher urgency = more points
  return Math.max(0, 1 - days / MAX_URGENCY_DAYS) * WEIGHTS.urgency;
}

export function computeScore(patient) {
  const days = daysUntil(patient.apptDate);

  const total =
    getNoShowRatePoints(patient.priorNoShows, patient.priorAppts) +
    getSocioeconomicPoints(patient.insurance) +
    getNewPatientPoints(patient.priorAppts) +
    getDistancePoints(patient.distanceMiles) +
    getUrgencyPoints(days);

  return Math.min(100, Math.round(total));
}

export function getRiskLevel(score) {
  if (score >= THRESHOLDS.HIGH) return "High";
  if (score >= THRESHOLDS.MEDIUM) return "Medium";
  return "Low";
}

export function buildExplanation(patient) {
  const days = daysUntil(patient.apptDate);
  const parts = [];

  if (patient.priorAppts === 0) {
    parts.push("new patient");
  } else {
    const rate = Math.round((patient.priorNoShows / patient.priorAppts) * 100);
    parts.push(`${patient.priorNoShows} of ${patient.priorAppts} prior appointments missed (${rate}%)`);
  }

  if (patient.distanceMiles >= 15) {
    parts.push(`${patient.distanceMiles} mi from clinic`);
  }

  if (days <= 3) {
    parts.push(`appointment in ${days} day${days === 1 ? "" : "s"} — act now`);
  } else {
    parts.push(`${days} days until appointment`);
  }

  return parts.join(" · ");
}

export function scoreAndSortPatients(patients) {
  return patients
    .map(p => {
      const score = computeScore(p);
      return { ...p, score, level: getRiskLevel(score) };
    })
    .sort((a, b) => b.score - a.score);
}
