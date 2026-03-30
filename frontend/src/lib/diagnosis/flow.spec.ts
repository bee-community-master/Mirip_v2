import { describe, expect, it } from 'vitest';

import { transitionDiagnosisStage } from '$lib/diagnosis/flow';

describe('transitionDiagnosisStage', () => {
	it('moves from upload to analyzing when a diagnosis starts', () => {
		expect(transitionDiagnosisStage('upload', { type: 'upload_started' })).toBe('analyzing');
	});

	it('moves from analyzing to result when the job succeeds with a result', () => {
		expect(
			transitionDiagnosisStage('analyzing', {
				type: 'job_polled',
				jobStatus: 'succeeded',
				hasResult: true
			})
		).toBe('result');
	});

	it('moves to error when the job fails', () => {
		expect(
			transitionDiagnosisStage('analyzing', {
				type: 'job_polled',
				jobStatus: 'failed',
				hasResult: false
			})
		).toBe('error');
	});
});
