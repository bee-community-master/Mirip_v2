export type TierKey = 'FREE' | 'STANDARD' | 'PRO';
export type UniversityKey = 'HONGIK' | 'KONKUK' | 'KOOKMIN';
export type ExpertTab = 'AI' | 'INSTRUCTOR' | 'PROFESSOR';
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

export interface ProbabilitySegment {
	label: string;
	value: number;
	fillClass: string;
	textClass: string;
}

export interface DiagnosisTierResult {
	predictedGrade: string;
	probability: number;
	confidence: number;
	segments: ProbabilitySegment[];
}

export interface AxisDistance {
	label: string;
	value: string;
	needImprovement: boolean;
}

export interface HistoryNote {
	label: string;
	tone: 'blue' | 'gold' | 'rose';
	text: string;
}

export interface HistoryRow {
	year: string;
	scores: number[];
	diffs: number[];
	isMine?: boolean;
}

export interface ExpertGuideItem {
	axis: string;
	status: string;
	needImprovement: boolean;
	text: string;
}

export interface UniversityAnalysis {
	key: UniversityKey;
	label: string;
	name: string;
	match: number;
	rank: string;
	stats: Array<{ label: string; value: string; accent?: boolean }>;
	axisDistances: AxisDistance[];
	strengths: string[];
	improvements: string[];
	historyNotes: HistoryNote[];
	historyRows: HistoryRow[];
	expertGuides: Record<ExpertTab, ExpertGuideItem[]>;
}

export interface DiagnosisMockData {
	radar: DiagnosisRadarPoint[];
	tierResult: DiagnosisTierResult;
	standardMatches: Array<{
		key: UniversityKey;
		name: string;
		match: number;
		rank: string;
		applicants: string;
		ratio: string;
	}>;
	universities: Record<UniversityKey, UniversityAnalysis>;
}
