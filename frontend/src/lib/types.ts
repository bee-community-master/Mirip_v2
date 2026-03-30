export type CompetitionCategory = '디지털' | '회화' | '캐릭터';
export type CompetitionFilter = '전체' | CompetitionCategory;

export interface MockUserSession {
	isAuthenticated: boolean;
	displayName: string;
	tierLabel: string;
	avatar: string;
	status: string;
	streak: number;
}

export interface HomeFeature {
	title: string;
	description: string;
	icon: 'trend' | 'sparkles' | 'trophy';
}

export interface CompetitionItem {
	id: string;
	title: string;
	org: string;
	dday: string;
	category: CompetitionCategory;
	tags: string[];
	image: string;
	prize: string;
	summary: string;
	deadline: string;
	region: string;
	participants: string;
	difficulty: string;
	matchingReason: string[];
}

export interface PortfolioProfile {
	name: string;
	avatar: string;
	role: string;
	stats: Array<{ label: string; value: string }>;
}

export interface PortfolioWork {
	id: string;
	title: string;
	image: string;
	likes: number;
	comments: number;
	height: 'medium' | 'tall';
	year: string;
	tags: string[];
	description: string;
}

export interface DiagnosisRadarPoint {
	subject: string;
	score: number;
	fullMark: number;
}

export interface DiagnosisProbabilityItem {
	university: string;
	department: string;
	probability: number;
	percentLabel: string;
}

export interface DiagnosisFeedbackSummary {
	overall: string | null;
	strengths: string[];
	improvements: string[];
}

export interface DiagnosisResultView {
	tier: string;
	summary: string | null;
	radarPoints: DiagnosisRadarPoint[];
	probabilities: DiagnosisProbabilityItem[];
	feedback: DiagnosisFeedbackSummary;
}
