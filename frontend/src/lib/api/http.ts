import { env } from '$env/dynamic/public';

import type { ApiErrorResponse } from '$lib/api/types';

const DEFAULT_API_BASE_URL = 'http://localhost:8000';

export class MiripApiError extends Error {
	status: number;
	code: string;
	detail: Record<string, unknown> | null | undefined;

	constructor({
		status,
		code,
		message,
		detail
	}: {
		status: number;
		code: string;
		message: string;
		detail?: Record<string, unknown> | null;
	}) {
		super(message);
		this.name = 'MiripApiError';
		this.status = status;
		this.code = code;
		this.detail = detail;
	}
}

export interface RequestJsonOptions {
	method?: 'GET' | 'POST' | 'PUT';
	body?: unknown;
	headers?: HeadersInit;
	fetchFn?: typeof fetch;
}

export interface JsonResponse<T> {
	data: T;
	headers: Headers;
}

function trimTrailingSlash(value: string) {
	return value.endsWith('/') ? value.slice(0, -1) : value;
}

function buildApiUrl(pathname: string) {
	const baseUrl = trimTrailingSlash(env.PUBLIC_MIRIP_API_BASE_URL || DEFAULT_API_BASE_URL);
	const path = pathname.startsWith('/') ? pathname : `/${pathname}`;
	return `${baseUrl}${path}`;
}

export async function requestJson<T>(
	pathname: string,
	{ method = 'GET', body, headers, fetchFn = fetch }: RequestJsonOptions = {}
): Promise<JsonResponse<T>> {
	const requestHeaders = new Headers(headers);

	if (body !== undefined && !requestHeaders.has('content-type')) {
		requestHeaders.set('content-type', 'application/json');
	}

	const response = await fetchFn(buildApiUrl(pathname), {
		method,
		headers: requestHeaders,
		body: body === undefined ? undefined : JSON.stringify(body)
	});

	if (!response.ok) {
		let payload: ApiErrorResponse | null = null;

		try {
			payload = (await response.json()) as ApiErrorResponse;
		} catch {
			payload = null;
		}

		throw new MiripApiError({
			status: response.status,
			code: payload?.code ?? 'request_failed',
			message: payload?.message ?? `Request failed with status ${response.status}`,
			detail: payload?.detail
		});
	}

	return {
		data: (await response.json()) as T,
		headers: response.headers
	};
}
