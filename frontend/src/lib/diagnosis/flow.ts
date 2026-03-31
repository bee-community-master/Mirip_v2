import type { DiagnosisJobStatus } from '$lib/api/types';

export type DiagnosisStage = 'upload' | 'analyzing' | 'result' | 'error';

export type DiagnosisFlowEvent =
	| { type: 'upload_started' }
	| { type: 'job_polled'; jobStatus: DiagnosisJobStatus; hasResult: boolean }
	| { type: 'request_failed' };

export function transitionDiagnosisStage(
	currentStage: DiagnosisStage,
	event: DiagnosisFlowEvent
): DiagnosisStage {
	switch (event.type) {
		case 'upload_started':
			return 'analyzing';
		case 'request_failed':
			return 'error';
		case 'job_polled':
			if (event.jobStatus === 'succeeded' && event.hasResult) {
				return 'result';
			}

			if (event.jobStatus === 'failed' || event.jobStatus === 'expired') {
				return 'error';
			}

			return currentStage === 'upload' ? 'analyzing' : currentStage;
		default:
			return currentStage;
	}
}
