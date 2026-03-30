import type { DiagnosisResultDto, DiagnosisResultProbabilityDto } from '$lib/api/types';
import type { DiagnosisFeedbackSummary, DiagnosisProbabilityItem, DiagnosisRadarPoint, DiagnosisResultView } from '$lib/types';

const SCORE_LABELS: Record<string, string> = {
	composition: '구성력',
	technique: '표현력',
	creativity: '창의성',
	completeness: '완성도'
};

const SCORE_ORDER = ['composition', 'technique', 'creativity', 'completeness'] as const;

function toRoundedScore(value: number) {
	return Number(value.toFixed(1));
}

function toPercentLabel(value: number) {
	return `${Math.round(value * 100)}%`;
}

function isStringArray(value: unknown): value is string[] {
	return Array.isArray(value) && value.every((item) => typeof item === 'string');
}

function readFeedbackSummary(feedback: Record<string, unknown> | null): DiagnosisFeedbackSummary {
	return {
		overall: typeof feedback?.overall === 'string' ? feedback.overall : null,
		strengths: isStringArray(feedback?.strengths) ? feedback.strengths : [],
		improvements: isStringArray(feedback?.improvements) ? feedback.improvements : []
	};
}

function mapScoreEntries(scores: Record<string, number>): DiagnosisRadarPoint[] {
	const orderedKeys = [
		...SCORE_ORDER.filter((key) => key in scores),
		...Object.keys(scores).filter((key) => !SCORE_ORDER.includes(key as (typeof SCORE_ORDER)[number]))
	];

	return orderedKeys.map((key) => ({
		subject: SCORE_LABELS[key] ?? key,
		score: toRoundedScore(scores[key]),
		fullMark: 100
	}));
}

function mapProbabilityEntry(entry: DiagnosisResultProbabilityDto): DiagnosisProbabilityItem | null {
	if (
		typeof entry.university !== 'string' ||
		typeof entry.department !== 'string' ||
		typeof entry.probability !== 'number'
	) {
		return null;
	}

	return {
		university: entry.university,
		department: entry.department,
		probability: entry.probability,
		percentLabel: toPercentLabel(entry.probability)
	};
}

export function mapDiagnosisResult(result: DiagnosisResultDto): DiagnosisResultView {
	return {
		tier: result.tier,
		summary: result.summary,
		radarPoints: mapScoreEntries(result.scores),
		probabilities: result.probabilities
			.map(mapProbabilityEntry)
			.filter((item): item is DiagnosisProbabilityItem => item !== null)
			.sort((left, right) => right.probability - left.probability),
		feedback: readFeedbackSummary(result.feedback)
	};
}
