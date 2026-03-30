export type QueryParamValue = string | null | undefined;

export function buildPathWithQuery(
	pathname: string,
	searchParams: URLSearchParams,
	updates: Record<string, QueryParamValue>
) {
	const nextSearchParams = new URLSearchParams(searchParams);

	for (const [key, value] of Object.entries(updates)) {
		if (value === null || value === undefined || value === '') {
			nextSearchParams.delete(key);
			continue;
		}

		nextSearchParams.set(key, value);
	}

	const query = nextSearchParams.toString();
	return query ? `${pathname}?${query}` : pathname;
}

export function readQueryOption<T extends string>(
	searchParams: URLSearchParams,
	key: string,
	options: readonly T[],
	fallback: T
) {
	const value = searchParams.get(key);
	return options.find((option) => option === value) ?? fallback;
}
