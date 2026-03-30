import { describe, expect, it } from 'vitest';

import { mapDiagnosisResult } from '$lib/diagnosis/adapter';

describe('mapDiagnosisResult', () => {
	it('maps backend scores, probabilities, and feedback into the diagnosis view model', () => {
		const result = mapDiagnosisResult({
			id: 'res-1',
			job_id: 'job-1',
			tier: 'A',
			scores: {
				composition: 82.4,
				technique: 78.1,
				creativity: 75.9,
				completeness: 81.7
			},
			probabilities: [
				{
					university: '홍익대학교',
					department: '시각디자인과',
					probability: 0.61
				},
				{
					university: '국민대학교',
					department: '공업디자인학과',
					probability: 0.48
				}
			],
			feedback: {
				overall: '시각디자인 기준 A 티어입니다.',
				strengths: ['구도 안정감이 좋습니다.'],
				improvements: ['표현 밀도를 더 높여보세요.']
			},
			summary: 'Single diagnosis completed.',
			created_at: '2026-03-31T10:00:00Z'
		});

		expect(result.radarPoints).toEqual([
			{ subject: '구성력', score: 82.4, fullMark: 100 },
			{ subject: '표현력', score: 78.1, fullMark: 100 },
			{ subject: '창의성', score: 75.9, fullMark: 100 },
			{ subject: '완성도', score: 81.7, fullMark: 100 }
		]);
		expect(result.probabilities[0]).toEqual({
			university: '홍익대학교',
			department: '시각디자인과',
			probability: 0.61,
			percentLabel: '61%'
		});
		expect(result.feedback).toEqual({
			overall: '시각디자인 기준 A 티어입니다.',
			strengths: ['구도 안정감이 좋습니다.'],
			improvements: ['표현 밀도를 더 높여보세요.']
		});
	});
});
