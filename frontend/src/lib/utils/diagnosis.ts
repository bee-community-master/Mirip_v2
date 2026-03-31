import { diagnosisMock } from '$lib/mocks/diagnosis';
import type {
	DiagnosisMockData,
	DiagnosisRadarPoint,
	DiagnosisTierResult,
	ProbabilitySegment,
	UniversityAnalysis,
	UniversityKey
} from '$lib/types';

const tierStyle: Record<string, { fillClass: string; textClass: string }> = {
	S: { fillClass: 'bg-yellow-400 text-black', textClass: 'text-black' },
	A: { fillClass: 'bg-white text-black', textClass: 'text-black' },
	B: { fillClass: 'bg-orange-400/18', textClass: 'text-orange-100' },
	C: { fillClass: 'bg-blue-500/18', textClass: 'text-blue-100' }
};

function hashSeed(seedInput: string): number {
	let hash = 2166136261;
	for (let index = 0; index < seedInput.length; index += 1) {
		hash ^= seedInput.charCodeAt(index);
		hash = Math.imul(hash, 16777619);
	}
	return hash >>> 0;
}

function mulberry32(seed: number) {
	let value = seed >>> 0;
	return () => {
		value += 0x6d2b79f5;
		let t = value;
		t = Math.imul(t ^ (t >>> 15), t | 1);
		t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
		return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
	};
}

function randomInt(rng: () => number, min: number, max: number): number {
	return Math.floor(rng() * (max - min + 1)) + min;
}

function clamp(value: number, min: number, max: number): number {
	return Math.min(max, Math.max(min, value));
}

function buildTierResult(radar: DiagnosisRadarPoint[], rng: () => number): DiagnosisTierResult {
	const average = radar.reduce((sum, point) => sum + point.score, 0) / radar.length;
	const predictedGrade = average >= 88 ? 'S' : average >= 79 ? 'A' : average >= 70 ? 'B' : 'C';
	const leadingProbability = clamp(randomInt(rng, 46, 71), 46, 71);
	const remaining = 100 - leadingProbability;
	const supportingOne = randomInt(rng, 8, Math.max(8, remaining - 10));
	const supportingTwo = randomInt(rng, 5, Math.max(5, remaining - supportingOne - 5));
	const trailing = 100 - leadingProbability - supportingOne - supportingTwo;
	const segmentsByTier: Record<string, number[]> = {
		S: [leadingProbability, supportingOne, supportingTwo, trailing],
		A: [trailing, leadingProbability, supportingOne, supportingTwo],
		B: [supportingTwo, supportingOne, leadingProbability, trailing],
		C: [supportingOne, trailing, supportingTwo, leadingProbability]
	};
	const orderedTiers = ['S', 'A', 'B', 'C'];
	const segments = orderedTiers.map((grade, index) => {
		const value = segmentsByTier[predictedGrade][index];
		const label = value >= 10 ? `${grade} ${value}%` : '';
		const style = tierStyle[grade];
		return {
			label,
			value,
			fillClass: style.fillClass,
			textClass: style.textClass
		} satisfies ProbabilitySegment;
	});
	return {
		predictedGrade,
		probability: leadingProbability,
		confidence: clamp(randomInt(rng, 82, 97), 82, 97),
		segments
	};
}

function updateUniversity(
	university: UniversityAnalysis,
	radar: DiagnosisRadarPoint[],
	rng: () => number
): UniversityAnalysis {
	const cutoffScores = university.historyRows[1]?.scores ?? radar.map((point) => point.score - 4);
	const scoreDiffs = radar.map((point, index) => point.score - cutoffScores[index]);
	const rankValue = randomInt(rng, 18, 48);
	const match = clamp(
		Math.round(
			72 +
				scoreDiffs.reduce((sum, diff) => sum + diff, 0) / scoreDiffs.length * 2.4 +
				rng() * 10
		),
		58,
		96
	);

	return {
		...university,
		match,
		rank: `상위 ${rankValue}%`,
		stats: university.stats.map((stat, index) =>
			index === 2 ? { ...stat, value: `상위 ${rankValue}%` } : stat
		),
		axisDistances: university.axisDistances.map((axis, index) => {
			const diff = scoreDiffs[index];
			return {
				...axis,
				value: diff >= 0 ? '합격권' : `+${Math.abs(diff)} 필요`,
				needImprovement: diff < 0
			};
		}),
		historyRows: university.historyRows.map((row, index) => {
			if (index === 0 || row.isMine) {
				return {
					...row,
					scores: radar.map((point) => point.score),
					diffs: radar.map(() => 0),
					isMine: true
				};
			}
			return {
				...row,
				diffs: radar.map((point, scoreIndex) => point.score - row.scores[scoreIndex])
			};
		}),
		expertGuides: Object.fromEntries(
			Object.entries(university.expertGuides).map(([expertKey, items]) => [
				expertKey,
				items.map((item, index) => {
					const diff = scoreDiffs[index % scoreDiffs.length];
					return {
						...item,
						status: diff >= 0 ? '합격권' : `+${Math.abs(diff)} 필요`,
						needImprovement: diff < 0
					};
				})
			])
		) as UniversityAnalysis['expertGuides']
	};
}

export function createDiagnosisStubData(seedInput: string): DiagnosisMockData {
	const rng = mulberry32(hashSeed(seedInput));
	const cloned = structuredClone(diagnosisMock) as DiagnosisMockData;
	const radar = cloned.radar.map((point, index) => ({
		...point,
		score: clamp(randomInt(rng, 64 + index * 2, 94), 58, 97)
	}));
	const tierResult = buildTierResult(radar, rng);
	const universityKeys = Object.keys(cloned.universities) as UniversityKey[];
	const universities = Object.fromEntries(
		universityKeys.map((key) => [
			key,
			updateUniversity(cloned.universities[key], radar, rng)
		])
	) as DiagnosisMockData['universities'];
	const standardMatches = cloned.standardMatches
		.map((match) => {
			const updated = universities[match.key];
			return {
				...match,
				match: updated.match,
				rank: updated.rank.replace('상위 ', ''),
				applicants: `${randomInt(rng, 860, 1480).toLocaleString('ko-KR')}명`,
				ratio: `${(randomInt(rng, 54, 118) / 10).toFixed(1)}:1`
			};
		})
		.sort((left, right) => right.match - left.match);

	return {
		radar,
		tierResult,
		standardMatches,
		universities
	};
}
