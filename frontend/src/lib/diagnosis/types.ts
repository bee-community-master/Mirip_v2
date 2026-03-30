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
