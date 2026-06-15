export const TODAY = new Date("2026-04-21");

export function daysUntil(dateStr) {
  const appt = new Date(dateStr);
  return Math.round((appt - TODAY) / (1000 * 60 * 60 * 24));
}

export function formatApptDate(dateStr) {
  return new Date(dateStr).toLocaleDateString("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  });
}

export function formatTodayLong() {
  return TODAY.toLocaleDateString("en-US", {
    weekday: "long",
    month: "long",
    day: "numeric",
    year: "numeric",
  });
}
