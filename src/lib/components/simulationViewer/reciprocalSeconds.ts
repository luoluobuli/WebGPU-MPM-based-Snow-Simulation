export const formatReciprocalSecondsDivisor = (value: number) =>
    `1 / ${value.toFixed(1)} s`;

export const parseReciprocalSecondsDivisor = (value: string) => {
    const trimmed = value.trim();
    const reciprocalMatch = /^1\s*\/\s*(\d+(?:\.\d+)?)\s*s?$/i.exec(trimmed);
    const numericText = reciprocalMatch?.[1] ?? trimmed.replace(/\s*s$/i, "");
    const parsed = Number(numericText);

    return Number.isFinite(parsed) ? parsed : null;
};
