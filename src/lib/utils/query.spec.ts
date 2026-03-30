import { describe, expect, it } from 'vitest';
import { buildPathWithQuery, readQueryOption } from '$lib/utils/query';

describe('buildPathWithQuery', () => {
	it('applies updates and removes nullish values', () => {
		const path = buildPathWithQuery('/competitions', new URLSearchParams('filter=디지털&item=abc'), {
			filter: null,
			item: 'next'
		});

		expect(path).toBe('/competitions?item=next');
	});

	it('returns the bare pathname when every query value is removed', () => {
		const path = buildPathWithQuery('/portfolio', new URLSearchParams('work=current'), {
			work: null
		});

		expect(path).toBe('/portfolio');
	});
});

describe('readQueryOption', () => {
	it('returns the matching query value when it is allowed', () => {
		const value = readQueryOption(
			new URLSearchParams('tier=PRO'),
			'tier',
			['FREE', 'STANDARD', 'PRO'] as const,
			'FREE'
		);

		expect(value).toBe('PRO');
	});

	it('falls back when the query value is missing or invalid', () => {
		const missingValue = readQueryOption(
			new URLSearchParams(),
			'tier',
			['FREE', 'STANDARD', 'PRO'] as const,
			'FREE'
		);
		const invalidValue = readQueryOption(
			new URLSearchParams('tier=VIP'),
			'tier',
			['FREE', 'STANDARD', 'PRO'] as const,
			'FREE'
		);

		expect(missingValue).toBe('FREE');
		expect(invalidValue).toBe('FREE');
	});
});
