export interface ApiErrorResponse {
	code: string;
	message: string;
	detail?: Record<string, unknown> | null;
}

export interface UploadAssetDto {
	id: string;
	filename: string;
	content_type: string;
	size_bytes: number;
	object_name: string;
	category: string | null;
	status: 'pending' | 'uploaded' | 'consumed';
	created_at: string;
}

export interface UploadSessionDto {
	upload_url: string;
	method: string;
	object_name: string;
	headers: Record<string, string>;
	expires_at: string | null;
}

export interface CreateUploadSessionResponseDto {
	upload: UploadAssetDto;
	session: UploadSessionDto;
}

export interface CompleteUploadResponseDto {
	upload: UploadAssetDto;
}

export type DiagnosisJobType = 'evaluate' | 'compare';
export type DiagnosisDepartment = 'visual_design' | 'industrial_design' | 'fine_art' | 'craft';
export type DiagnosisLanguage = 'ko' | 'en';
export type DiagnosisJobStatus =
	| 'queued'
	| 'leased'
	| 'running'
	| 'succeeded'
	| 'failed'
	| 'expired';

export interface CreateDiagnosisJobRequestDto {
	upload_ids: string[];
	job_type: DiagnosisJobType;
	department: DiagnosisDepartment;
	include_feedback: boolean;
	theme: string | null;
	language: DiagnosisLanguage;
}

export interface DiagnosisJobDto {
	id: string;
	job_type: string;
	department: string;
	status: DiagnosisJobStatus;
	upload_ids: string[];
	created_at: string;
	updated_at: string;
	attempts: number;
	failure_reason: string | null;
}

export interface DiagnosisResultProbabilityDto {
	university?: unknown;
	department?: unknown;
	probability?: unknown;
	[key: string]: unknown;
}

export interface DiagnosisResultDto {
	id: string;
	job_id: string;
	tier: string;
	scores: Record<string, number>;
	probabilities: DiagnosisResultProbabilityDto[];
	feedback: Record<string, unknown> | null;
	summary: string | null;
	created_at: string;
}

export interface DiagnosisJobStatusResponseDto {
	job: DiagnosisJobDto;
	result: DiagnosisResultDto | null;
}
