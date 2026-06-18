import { describe, expect, it } from "vitest";

import {
    formatReciprocalSecondsDivisor,
    parseReciprocalSecondsDivisor,
} from "./reciprocalSeconds";

describe("reciprocal seconds entry values", () => {
    it("formats and parses reciprocal-second divisors", () => {
        expect(formatReciprocalSecondsDivisor(30)).toBe("1 / 30.0 s");
        expect(parseReciprocalSecondsDivisor("1 / 30 s")).toBe(30);
        expect(parseReciprocalSecondsDivisor("30")).toBe(30);
        expect(parseReciprocalSecondsDivisor("30 s")).toBe(30);
        expect(parseReciprocalSecondsDivisor("nope")).toBeNull();
    });
});
