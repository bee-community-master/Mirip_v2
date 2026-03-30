import { describe, expect, it } from 'vitest';

import {
	diagnosisPollIntervalInMs,
	getDiagnosisFailureMessage,
	getDiagnosisJobStatusLabel,
	validateDiagnosisFile
} from '$lib/diagnosis/logic';

describe('diagnosis logic', () => {
	it('keeps the polling interval in one place', () => {
		expect(diagnosisPollIntervalInMs).toBe(2500);
	});

	it('rejects unsupported files with a clear validation message', () => {
		expect(() =>
			validateDiagnosisFile({
				type: 'application/pdf',
				size: 1024
			})
		).toThrow('PNG, JPG, JPEG, WebP 형식의 이미지만 업로드할 수 있습니다.');
	});

	it('returns localized job labels and fallback failure copy', () => {
		expect(getDiagnosisJobStatusLabel('running')).toBe('분석 중');
		expect(getDiagnosisFailureMessage({ failure_reason: null })).toBe(
			'진단 작업이 완료되지 않았습니다. 로컬 worker 로그를 확인해 주세요.'
		);
	});
});
