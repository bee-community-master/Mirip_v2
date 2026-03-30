import { describe, expect, it } from 'vitest';
import { competitionItems } from '$lib/mocks/competitions';
import { filterCompetitions } from '$lib/utils/competition';

describe('filterCompetitions', () => {
	it('returns every item for the 전체 filter', () => {
		const results = filterCompetitions(competitionItems, '전체');

		expect(results).toHaveLength(competitionItems.length);
	});

	it('returns only entries in the selected category', () => {
		const results = filterCompetitions(competitionItems, '디지털');

		expect(results.length).toBeGreaterThan(0);
		expect(results.every((item) => item.category === '디지털')).toBe(true);
	});
});
