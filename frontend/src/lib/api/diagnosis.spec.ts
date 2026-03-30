import { describe, expect, it } from 'vitest';

import { isFakeUploadMode } from '$lib/api/diagnosis';

describe('isFakeUploadMode', () => {
	it('returns true when the API response header marks fake mode', () => {
		expect(
			isFakeUploadMode({ headers: {} }, new Headers({ 'x-mirip-mode': 'fake' }))
		).toBe(true);
	});

	it('returns true when the signed upload session headers mark fake mode', () => {
		expect(
			isFakeUploadMode({
				headers: {
					'x-mirip-mode': 'fake'
				}
			})
		).toBe(true);
	});

	it('returns false when neither response nor session headers mark fake mode', () => {
		expect(
			isFakeUploadMode(
				{
					headers: {
						'content-type': 'image/png'
					}
				},
				new Headers()
			)
		).toBe(false);
	});
});
