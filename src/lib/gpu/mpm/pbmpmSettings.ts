export const DEFAULT_PBMPM_SOLVE_ITERATIONS = 4;
export const MAX_PBMPM_SOLVE_ITERATIONS = 32;

export const normalizePbmpmSolveIterations = (iterations: number) => {
    return Number.isFinite(iterations)
        ? Math.max(1, Math.min(MAX_PBMPM_SOLVE_ITERATIONS, Math.round(iterations)))
        : DEFAULT_PBMPM_SOLVE_ITERATIONS;
};
