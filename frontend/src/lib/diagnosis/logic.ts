import type { DiagnosisJobDto, DiagnosisJobStatus } from '$lib/api/types';

export interface DiagnosisFileLike {
	type: string;
	size: number;
}

const validImageTypes = new Set(['image/png', 'image/jpeg', 'image/jpg', 'image/webp']);
const maxUploadSizeInBytes = 10 * 1024 * 1024;

export const diagnosisPollIntervalInMs = 2500;

const diagnosisJobStatusLabels: Record<DiagnosisJobStatus, string> = {
	queued: '대기 중',
	leased: '워커 할당 중',
	running: '분석 중',
	succeeded: '완료',
	failed: '실패',
	expired: '만료'
};

export function validateDiagnosisFile(file: DiagnosisFileLike) {
	if (!validImageTypes.has(file.type)) {
		throw new Error('PNG, JPG, JPEG, WebP 형식의 이미지만 업로드할 수 있습니다.');
	}

	if (file.size > maxUploadSizeInBytes) {
		throw new Error('파일 크기는 10 MB 이하로 올려주세요.');
	}
}

export function getDiagnosisJobStatusLabel(status: DiagnosisJobStatus) {
	return diagnosisJobStatusLabels[status];
}

export function getDiagnosisFailureMessage(job: Pick<DiagnosisJobDto, 'failure_reason'>) {
	return job.failure_reason ?? '진단 작업이 완료되지 않았습니다. 로컬 worker 로그를 확인해 주세요.';
}
