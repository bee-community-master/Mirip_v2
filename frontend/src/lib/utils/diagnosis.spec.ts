import { describe, expect, it } from 'vitest';

import { createDiagnosisStubData } from '$lib/utils/diagnosis';

describe('createDiagnosisStubData', () => {
	it('returns deterministic stub data for the same seed', () => {
		const first = createDiagnosisStubData('same-seed');
		const second = createDiagnosisStubData('same-seed');

		expect(first).toEqual(second);
	});

	it('returns plausible tier segment totals', () => {
		const data = createDiagnosisStubData('totals-seed');
		const total = data.tierResult.segments.reduce((sum, segment) => sum + segment.value, 0);

		expect(total).toBe(100);
		expect(data.radar.every((point) => point.score >= 58 && point.score <= 97)).toBe(true);
	});
});
