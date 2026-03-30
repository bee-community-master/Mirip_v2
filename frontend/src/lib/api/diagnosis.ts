import { MiripApiError, requestJson, type JsonResponse } from '$lib/api/http';
import type {
	CompleteUploadResponseDto,
	CreateDiagnosisJobRequestDto,
	CreateUploadSessionResponseDto,
	DiagnosisJobDto,
	DiagnosisJobStatusResponseDto,
	UploadSessionDto
} from '$lib/api/types';

export function isFakeUploadMode(
	session: Pick<UploadSessionDto, 'headers'>,
	responseHeaders?: Headers
): boolean {
	const headerValue = responseHeaders?.get('x-mirip-mode');
	const sessionValue =
		session.headers['x-mirip-mode'] ?? session.headers['X-Mirip-Mode'] ?? session.headers['X-MIRIP-MODE'];

	return headerValue === 'fake' || sessionValue === 'fake';
}

export async function createUploadSession(
	file: File,
	fetchFn?: typeof fetch
): Promise<JsonResponse<CreateUploadSessionResponseDto>> {
	return requestJson<CreateUploadSessionResponseDto>('/v1/uploads', {
		method: 'POST',
		body: {
			filename: file.name,
			content_type: file.type,
			size_bytes: file.size,
			category: 'diagnosis'
		},
		fetchFn
	});
}

export async function uploadToSignedUrl(
	session: UploadSessionDto,
	file: File,
	fetchFn: typeof fetch = fetch
): Promise<void> {
	const uploadHeaders = new Headers();

	for (const [key, value] of Object.entries(session.headers)) {
		if (key.toLowerCase() === 'x-mirip-mode') {
			continue;
		}

		uploadHeaders.set(key, value);
	}

	if (!uploadHeaders.has('content-type')) {
		uploadHeaders.set('content-type', file.type);
	}

	const response = await fetchFn(session.upload_url, {
		method: session.method,
		headers: uploadHeaders,
		body: file
	});

	if (!response.ok) {
		throw new MiripApiError({
			status: response.status,
			code: 'upload_failed',
			message: 'Signed upload failed.'
		});
	}
}

export async function completeUpload(
	uploadId: string,
	fetchFn?: typeof fetch
): Promise<CompleteUploadResponseDto> {
	const response = await requestJson<CompleteUploadResponseDto>(`/v1/uploads/${uploadId}/complete`, {
		method: 'POST',
		fetchFn
	});

	return response.data;
}

export async function createDiagnosisJob(
	payload: CreateDiagnosisJobRequestDto,
	fetchFn?: typeof fetch
): Promise<DiagnosisJobDto> {
	const response = await requestJson<DiagnosisJobDto>('/v1/diagnosis/jobs', {
		method: 'POST',
		body: payload,
		fetchFn
	});

	return response.data;
}

export async function getDiagnosisJobStatus(
	jobId: string,
	fetchFn?: typeof fetch
): Promise<DiagnosisJobStatusResponseDto> {
	const response = await requestJson<DiagnosisJobStatusResponseDto>(`/v1/diagnosis/jobs/${jobId}`, {
		fetchFn
	});

	return response.data;
}
